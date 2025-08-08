# Copyright 2025 Igor Bogdanov
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import openai
import json
import requests
import time
from typing import List, Dict, Any, Optional, Tuple
from groq import Groq
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import atexit
import gc
import resource
import logging

# Import the new logger setup function
from .logger_config import setup_timestamped_logging

# Import settings from the new settings file
try:
    from .connector_settings import (
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
    OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
    LOCAL_LLAMA_BASE_URL = os.environ.get("LOCAL_LLAMA_BASE_URL")
    OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL")
    OPENROUTER_REFERER = os.environ.get("OPENROUTER_REFERER")
    OPENROUTER_SITE_NAME = os.environ.get("OPENROUTER_SITE_NAME")
    OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL")
    DEFAULT_PROVIDER = os.environ.get("DEFAULT_PROVIDER", "openrouter")
    DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "google/gemini-pro")
    MODEL_PRICING = {}

# --- Global State ---
_session = None
_openai_clients = {}
_groq_client = None
logger = logging.getLogger("LLMConnector")  # Placeholder logger
_session_stats = {}  # To track tokens and costs per provider
_logging_initialized = False  # Flag to ensure logging is set up only once


def _update_stats(provider, prompt_tokens, completion_tokens):
    """Update and log the session stats for a given provider."""
    provider_name, model_name = provider

    if provider_name not in _session_stats:
        _session_stats[provider_name] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cost": 0.0,
        }

    _session_stats[provider_name]["prompt_tokens"] += prompt_tokens
    _session_stats[provider_name]["completion_tokens"] += completion_tokens

    cost = 0.0
    price_key = (provider_name, model_name)
    if MODEL_PRICING and price_key in MODEL_PRICING:
        input_price, output_price = MODEL_PRICING[price_key]
        cost = ((prompt_tokens / 1_000_000) * input_price) + (
            (completion_tokens / 1_000_000) * output_price
        )
        _session_stats[provider_name]["cost"] += cost
        logger.info(
            f"Usage - Provider: {provider_name}, Model: {model_name}, Prompt: {prompt_tokens}, Completion: {completion_tokens}, Cost: ${cost:.6f}"
        )
    else:
        logger.info(
            f"Usage - Provider: {provider_name}, Model: {model_name}, Prompt: {prompt_tokens}, Completion: {completion_tokens}, Cost: (pricing not available)"
        )


def _initialize_session_and_logging(debug_mode=False):
    """
    Initializes the session and the timestamped logger.
    This function is called only once on the first chat_completion call.
    """
    global _session, logger, _logging_initialized
    if _logging_initialized:
        return

    # 1. Set up the timestamped logger for this run
    log_level = logging.DEBUG if debug_mode else logging.INFO
    logger = setup_timestamped_logging(log_level)

    # 2. Create the persistent requests session (moved from get_session)
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
    logger.info("Created new persistent session with connection pooling.")

    # 3. Configure system resources
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        new_soft = min(4096, hard)
        if new_soft > soft:
            resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
            logger.info(f"Increased file descriptor limit from {soft} to {new_soft}.")
        else:
            logger.info(f"Current file descriptor limit: {soft}.")
    except Exception as exception:
        logger.warning(f"Failed to adjust file descriptor limit: {exception}")

    _logging_initialized = True


def get_session():
    """Get a persistent session. Note: Initialization is now in chat_completion."""
    # The session is now guaranteed to be initialized by the time this is called.
    return _session


def cleanup_resources():
    """Clean up resources and log the session summary."""
    global _session, _openai_clients, _groq_client, _session_stats

    if _session_stats:
        summary_header = "\n--- LLM Connector Session Summary ---\n"
        summary_body = "{:<15} | {:>15} | {:>15} | {:>15}\n".format(
            "Provider", "Prompt Tokens", "Completion Tokens", "Total Cost"
        )
        summary_line = "-" * 67 + "\n"

        for provider, stats in _session_stats.items():
            cost_str = f"${stats['cost']:.6f}" if stats["cost"] > 0 else "N/A"
            summary_body += "{:<15} | {:>15,} | {:>15,} | {:>15}\n".format(
                provider, stats["prompt_tokens"], stats["completion_tokens"], cost_str
            )

        logger.info(summary_header + summary_body + summary_line)

    if _session:
        logger.info("Cleaning up global session")
        _session.close()
        _session = None
    _openai_clients.clear()
    _groq_client = None
    gc.collect()
    logger.info("All network resources cleaned up")


atexit.register(cleanup_resources)

# --- Client Configurations ---
LOCAL_LLAMA_CONFIG = {"api_key": "not-needed", "base_url": LOCAL_LLAMA_BASE_URL}
OLLAMA_CONFIG = {"api_key": "ollama", "base_url": OLLAMA_BASE_URL}


def get_client(provider: tuple[str, str]):
    """Get a client configured for the specified provider."""
    global _openai_clients, _groq_client
    provider_name, _ = provider
    if provider_name in ["local", "ollama"]:
        if provider_name not in _openai_clients:
            config = LOCAL_LLAMA_CONFIG if provider_name == "local" else OLLAMA_CONFIG
            _openai_clients[provider_name] = openai.OpenAI(**config)
            logger.info(f"Created new {provider_name} OpenAI client")
        return _openai_clients[provider_name]
    elif provider_name == "groq":
        if _groq_client is None:
            if not GROQ_API_KEY:
                raise ValueError("GROQ_API_KEY is not set.")
            _groq_client = Groq(api_key=GROQ_API_KEY)
            logger.info("Created new Groq client")
        return _groq_client
    elif provider_name == "openrouter":
        return None
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def openrouter_chat_completion(messages, model, temperature, max_tokens, top_p):
    """Send a chat completion request to OpenRouter."""
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY is not set.")
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
    request_url = f"{OPENROUTER_BASE_URL}/chat/completions"

    response_text, prompt_tokens, completion_tokens, total_tokens, latency = (
        None,
        0,
        0,
        0,
        0.0,
    )
    start_time = time.monotonic()

    try:
        session = get_session()
        logger.info(f"HTTP Request: POST {request_url} for model {model}")
        response = session.post(
            request_url, headers=headers, json=payload, timeout=(3.05, 60)
        )
        logger.info(f"HTTP Response: {response.status_code} {response.reason}")
        response.raise_for_status()
        response_data = response.json()
        if "choices" in response_data and response_data["choices"]:
            response_text = response_data["choices"][0]["message"]["content"]
            logger.debug(f"LLM Response: {response_text}")
            usage = response_data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)
            _update_stats(("openrouter", model), prompt_tokens, completion_tokens)
        else:
            response_text = "Error: Unexpected response format from OpenRouter"
            logger.error(f"{response_text}: {response_data}")
    except requests.exceptions.RequestException as exception:
        response_text = (
            f"Error with OpenRouter API: {type(exception).__name__}: {exception}"
        )
        logger.error(response_text)
        if hasattr(exception, "response") and exception.response:
            logger.error(f"Response status: {exception.response.status_code}")
            logger.error(f"Response text: {exception.response.text}")
    finally:
        latency = time.monotonic() - start_time
        if "response" in locals() and response:
            response.close()

    return response_text, prompt_tokens, completion_tokens, total_tokens, latency


def chat_completion(
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    max_tokens: int = 1024,
    provider: tuple[str, str] = (DEFAULT_PROVIDER, DEFAULT_MODEL),
    top_p: float = 0.7,
    debug: bool = True,
) -> Tuple[str, int, int, int, float]:
    """Generate a chat completion using the specified provider."""
    # Ensure session and logging are initialized. This only runs once.
    _initialize_session_and_logging(debug_mode=debug)

    response_text, prompt_tokens, completion_tokens, total_tokens, latency = (
        "Error: Init failed",
        0,
        0,
        0,
        0.0,
    )
    start_time = time.monotonic()

    try:
        provider_name, model_name = provider
        # With verbose logging enabled, log the full prompt
        logger.debug(
            f"Provider: {provider}, Messages: {json.dumps(messages, indent=2)}"
        )

        if provider_name == "openrouter":
            return openrouter_chat_completion(
                messages, model_name, temperature, max_tokens, top_p
            )

        client = get_client(provider)
        logger.info(
            f"Requesting completion from {provider_name} with model {model_name}"
        )
        start_call_time = time.monotonic()
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        latency = time.monotonic() - start_call_time

        if response and response.choices:
            response_text = response.choices[0].message.content
            logger.debug(f"LLM Response: {response_text}")
            if response.usage:
                prompt_tokens = response.usage.prompt_tokens or 0
                completion_tokens = response.usage.completion_tokens or 0
                total_tokens = response.usage.total_tokens or 0
                _update_stats(provider, prompt_tokens, completion_tokens)
        else:
            response_text = "Error: No response/choices received from API"

    except Exception as exception:
        error_type = type(exception).__name__
        error_message = str(exception)
        response_text = f"Error in chat_completion: {error_type}: {error_message}"
        logger.error(response_text)
        if "Too many open files" in str(exception):
            logger.warning(
                "Detected 'Too many open files' error, performing emergency cleanup"
            )
            cleanup_resources()
            gc.collect()

    if latency == 0.0:
        latency = time.monotonic() - start_time

    return response_text, prompt_tokens, completion_tokens, total_tokens, latency
