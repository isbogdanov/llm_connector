import os
import openai
import json
import logging
import requests
import time  # Import time for latency measurement
from typing import List, Dict, Any, Optional, Tuple  # Added Tuple
from groq import Groq
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import atexit
import gc
import resource

# Import settings from the new settings file
try:
    from connector_settings import (
        OPENROUTER_API_KEY,
        GROQ_API_KEY,
        LOCAL_LLAMA_BASE_URL,
        OLLAMA_BASE_URL,
        OPENROUTER_REFERER,
        OPENROUTER_SITE_NAME,
        OPENROUTER_BASE_URL,
        DEFAULT_PROVIDER,
        DEFAULT_MODEL,
        MODEL_PRICING,
    )
except ImportError:
    print(
        "Could not import from connector_settings.py, using fallback environment variables."
    )
    # Fallback to environment variables if the settings file is not found
    # This makes the connector more robust for different deployment scenarios
    OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
    LOCAL_LLAMA_BASE_URL = os.environ.get("LOCAL_LLAMA_BASE_URL")
    OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL")
    OPENROUTER_REFERER = os.environ.get("OPENROUTER_REFERER")
    OPENROUTER_SITE_NAME = os.environ.get("OPENROUTER_SITE_NAME")
    OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL")
    DEFAULT_PROVIDER = os.environ.get("DEFAULT_PROVIDER", "openrouter")
    DEFAULT_MODEL = os.environ.get(
        "DEFAULT_MODEL", "google/gemini-pro"
    )  # Provide a default
    MODEL_PRICING = {}


# Configure more detailed logging
logging.basicConfig(level=logging.INFO)  # Switch to DEBUG level
logger = logging.getLogger("LLMConnector")

# Set a higher file descriptor limit if possible
try:
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    new_soft = min(4096, hard)  # Try to increase to 4096 but don't exceed hard limit
    if new_soft > soft:
        resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
        logger.info(f"Increased file descriptor limit from {soft} to {new_soft}")
    else:
        logger.info(f"Current file descriptor limit: {soft}")
except Exception as e:
    logger.warning(f"Failed to adjust file descriptor limit: {e}")

# Client configurations are now built dynamically from settings
LOCAL_LLAMA_CONFIG = {
    "api_key": "not-needed",
    "base_url": LOCAL_LLAMA_BASE_URL,
}

OLLAMA_CONFIG = {
    "api_key": "ollama",  # Ollama requires a non-empty key, 'ollama' is standard
    "base_url": OLLAMA_BASE_URL,
}

# Create global session with connection pooling and retry logic
_session = None
_openai_clients = {}  # Cache for OpenAI clients
_groq_client = None


def get_session():
    """Get a persistent session with connection pooling"""
    global _session
    if _session is None:
        _session = requests.Session()
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "POST"],
            backoff_factor=0.5,
        )
        adapter = HTTPAdapter(
            pool_connections=10, pool_maxsize=110, max_retries=retry_strategy
        )
        _session.mount("http://", adapter)
        _session.mount("https://", adapter)
        logger.info("Created new persistent session with connection pooling")
    return _session


def cleanup_resources():
    """Clean up all global resources"""
    global _session, _openai_clients, _groq_client

    # Close the global session
    if _session is not None:
        logger.info("Cleaning up global session")
        _session.close()
        _session = None

    # Clear OpenAI client cache
    _openai_clients.clear()

    # Clean up Groq client
    _groq_client = None

    # Force garbage collection
    gc.collect()

    logger.info("All network resources cleaned up")


# Register cleanup to happen at program exit
atexit.register(cleanup_resources)


def get_client(provider: tuple[str, str]):
    """
    Get a client configured for the specified provider

    Args:
        provider (tuple): ('provider_name', 'model_name')

    Returns:
        Client instance (OpenAI for local/ollama, Groq for groq, None for openrouter)
    """
    global _openai_clients, _groq_client

    provider_name = provider[0]

    if provider_name in ["local", "ollama"]:
        # Cache OpenAI clients by provider to avoid creating too many
        if provider_name not in _openai_clients:
            if provider_name == "local":
                _openai_clients[provider_name] = openai.OpenAI(**LOCAL_LLAMA_CONFIG)
                logger.info(f"Created new {provider_name} OpenAI client")
            else:  # ollama
                # The base_url is now direct from settings
                _openai_clients[provider_name] = openai.OpenAI(**OLLAMA_CONFIG)
                logger.info(f"Created new {provider_name} OpenAI client")
        return _openai_clients[provider_name]

    elif provider_name == "groq":
        if _groq_client is None:
            if not GROQ_API_KEY:
                raise ValueError("GROQ_API_KEY is not set in settings or environment.")
            _groq_client = Groq(api_key=GROQ_API_KEY)
            logger.info("Created new Groq client")
        return _groq_client

    elif provider_name == "openrouter":
        # OpenRouter doesn't use a client instance; we use direct HTTP requests
        return None
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def openrouter_chat_completion(messages, model, temperature, max_tokens, top_p):
    """
    Send a chat completion request to OpenRouter

    Args:
        messages: List of message dictionaries
        model: Model name for OpenRouter
        temperature: Temperature parameter
        max_tokens: Maximum tokens in response
        top_p: Top-p sampling parameter

    Returns:
        Tuple[Optional[str], int, int, int, float]:
            (response_text, prompt_tokens, completion_tokens, total_tokens, latency)
    """
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY is not set in settings or environment.")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": OPENROUTER_REFERER,
        "X-Title": OPENROUTER_SITE_NAME,
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
    }

    # The base_url is now also from settings
    request_url = f"{OPENROUTER_BASE_URL}/chat/completions"

    response_text = None
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    latency = 0.0

    start_time = time.monotonic()
    try:
        # Log the request (similar to httpx format)
        logger.info(f"HTTP Request: POST {request_url}")

        # Use the persistent session with connection pooling
        session = get_session()
        response = session.post(
            request_url,
            headers=headers,
            json=payload,
            timeout=(3.05, 60),  # connect timeout, read timeout
        )

        # Log the response status
        logger.info(f"HTTP Response: {response.status_code} {response.reason}")

        try:
            response.raise_for_status()
            response_data = response.json()

            if "choices" in response_data and len(response_data["choices"]) > 0:
                response_text = response_data["choices"][0]["message"]["content"]
                # Extract usage info if available
                usage = response_data.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                total_tokens = usage.get("total_tokens", 0)
            else:
                logger.error(
                    f"Unexpected response format from OpenRouter: {response_data}"
                )
                response_text = "Error: Unexpected response format from OpenRouter"
        finally:
            # Always close the response to release the connection back to the pool
            response.close()

    except requests.exceptions.RequestException as e:
        logger.error(f"OpenRouter API error: {str(e)}")
        if hasattr(e, "response") and e.response:
            logger.error(f"Response status: {e.response.status_code}")
            logger.error(f"Response text: {e.response.text}")
            e.response.close()  # Close the error response too
        response_text = f"Error with OpenRouter API: {str(e)}"
    finally:
        latency = time.monotonic() - start_time

    return response_text, prompt_tokens, completion_tokens, total_tokens, latency


def chat_completion(
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    max_tokens: int = 1024,
    provider: tuple[str, str] = (DEFAULT_PROVIDER, DEFAULT_MODEL),
    top_p: float = 0.7,
    debug: bool = False,
) -> Tuple[str, int, int, int, float]:
    """
    Generate a chat completion using the specified provider

    Args:
        messages: List of message dictionaries
        temperature: Temperature parameter
        max_tokens: Maximum tokens in response
        provider: Tuple of (provider_name, model_name)
        top_p: Top-p sampling parameter
        debug: Enable extra debug output

    Returns:
        Tuple[str, int, int, int, float]:
            (response_text, prompt_tokens, completion_tokens, total_tokens, latency)
            Returns error message as text and 0s for tokens/latency on failure.
    """
    response_text = "Error: Initialization failed"
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    latency = 0.0
    start_time = time.monotonic()

    try:
        if debug:
            print(f"\n{'='*40}\nDEBUG MODE ENABLED\n{'='*40}")
            print(f"Provider: {provider}")
            print(f"Messages: {json.dumps(messages, indent=2)}")

        provider_name = provider[0]
        model_name = provider[1]

        # Handle OpenRouter separately since it doesn't use a client
        if provider_name == "openrouter":
            return openrouter_chat_completion(
                messages=messages,
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )

        # For other providers, get the appropriate client
        client = get_client(provider)
        response = None  # Initialize response variable

        # Make the API call with the appropriate client
        try:
            start_call_time = time.monotonic()
            if provider_name == "local":
                # For local llama.cpp
                response = client.chat.completions.create(
                    model=model_name,  # This doesn't matter for llama.cpp
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            elif provider_name == "ollama":
                # For Ollama
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            else:  # groq
                # For Groq API using native client
                response = client.chat.completions.create(
                    messages=messages,
                    model=model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                )
            call_latency = time.monotonic() - start_call_time
            latency = call_latency  # Primarily measure the API call time

            # Extract results and usage
            if response and response.choices:
                response_text = response.choices[0].message.content
                if response.usage:
                    prompt_tokens = response.usage.prompt_tokens or 0
                    completion_tokens = response.usage.completion_tokens or 0
                    total_tokens = response.usage.total_tokens or 0
                else:
                    # Handle cases where usage might be missing (e.g., older Ollama?)
                    response_text = response.choices[
                        0
                    ].message.content  # Still get content
                    # Tokens remain 0
            else:
                response_text = "Error: No response/choices received from API"
                # Tokens remain 0

            if debug:
                print(f"Success! Response received.")
                print(f"Response preview: {response_text[:100]}...")

            return (
                response_text,
                prompt_tokens,
                completion_tokens,
                total_tokens,
                latency,
            )

        except Exception as api_err:
            if debug:
                print(f"API call failed: {str(api_err)}")
                # Try to get more info from the exception
                err_details = str(api_err)
                if hasattr(api_err, "response"):
                    try:
                        err_details += f"\nStatus code: {api_err.response.status_code}"
                        err_details += f"\nResponse text: {api_err.response.text}"
                    except:
                        pass
                print(f"Error details: {err_details}")

            # Try forcing garbage collection as a last resort
            gc.collect()

            raise api_err

    except Exception as e:
        response_text = f"Error in chat_completion: {str(e)}"  # Update error message
        logger.error(response_text)
        # Reset tokens/latency on error
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        latency = time.monotonic() - start_time  # Record total time until error

        # If it's a "Too many open files" error, attempt emergency cleanup
        if "Too many open files" in str(e):
            logger.warning(
                "Detected 'Too many open files' error, performing emergency cleanup"
            )
            cleanup_resources()
            gc.collect()

    if latency == 0.0:
        latency = time.monotonic() - start_time

    return response_text, prompt_tokens, completion_tokens, total_tokens, latency
