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
import json
import requests
import time
from typing import List, Dict, Any, Optional, Tuple
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import atexit
import gc
import resource
import logging



from .helpers.logger_config import setup_timestamped_logging

from .helpers.llm_settings import (
    DEFAULT_PROVIDER,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TOP_P,
)
from .helpers.security_settings import (
    RETRY_MAX,
    RETRY_BACKOFF,
    RETRY_STATUSES,
    POOL_CONNECTIONS,
    POOL_MAXSIZE,
    RLIMIT_NOFILE,
)
from .helpers.pricing import pricing_manager

from .adapters import (
    GoogleAdapter,
    OpenRouterAdapter,
    LocalAdapter,
    GroqAdapter,
    OpenAIAdapter,
    AnthropicAdapter,
    VertexAdapter,
)

# --- Global State ---
_session = None
logger = logging.getLogger("LLMConnector")
_session_stats = {}
_logging_initialized = False

_adapters = {}

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

    input_price, output_price = pricing_manager.get_model_pricing(provider_name, model_name)
    has_pricing = pricing_manager.has_model_pricing(provider_name, model_name)
    cost = ((prompt_tokens / 1_000_000) * input_price) + (
        (completion_tokens / 1_000_000) * output_price
    )
    _session_stats[provider_name]["cost"] += cost
    
    if has_pricing:
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
    """
    global _session, logger, _logging_initialized
    if _logging_initialized:
        return

    # 1. Set up the timestamped logger for this run
    log_level = logging.DEBUG if debug_mode else logging.INFO
    logger = setup_timestamped_logging(log_level)

    # 2. Create the persistent requests session
    _session = requests.Session()
    retry_strategy = Retry(
        total=RETRY_MAX,
        status_forcelist=RETRY_STATUSES,
        allowed_methods=["HEAD", "GET", "POST"],
        backoff_factor=RETRY_BACKOFF,
    )
    adapter = HTTPAdapter(
        pool_connections=POOL_CONNECTIONS, pool_maxsize=POOL_MAXSIZE, max_retries=retry_strategy
    )
    _session.mount("http://", adapter)
    _session.mount("https://", adapter)
    logger.info("Created new persistent session with connection pooling.")

    # 3. Configure system resources
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        new_soft = min(RLIMIT_NOFILE, hard)
        if new_soft > soft:
            resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
            logger.info(f"Increased file descriptor limit from {soft} to {new_soft}.")
        else:
            logger.info(f"Current file descriptor limit: {soft}.")
    except Exception as exception:
        logger.warning(f"Failed to adjust file descriptor limit: {exception}")

    _logging_initialized = True


def get_session():
    """Get a persistent session for generic HTTP clients."""
    return _session


def get_adapter(provider_name: str):
    """Retrieve or instantiate the requested adapter."""
    global _adapters
    if provider_name not in _adapters:
        if provider_name == "google":
            _adapters[provider_name] = GoogleAdapter()
        elif provider_name == "openrouter":
            _adapters[provider_name] = OpenRouterAdapter()
        elif provider_name in ["local", "ollama"]:
            _adapters[provider_name] = LocalAdapter(provider_name)
        elif provider_name == "groq":
            _adapters[provider_name] = GroqAdapter()
        elif provider_name == "openai":
            _adapters[provider_name] = OpenAIAdapter()
        elif provider_name == "anthropic":
            _adapters[provider_name] = AnthropicAdapter()
        elif provider_name == "vertex":
            _adapters[provider_name] = VertexAdapter()
        else:
            raise ValueError(f"Unsupported provider: {provider_name}")
    return _adapters[provider_name]


def cleanup_resources():
    """Clean up resources and log the session summary."""
    global _session, _session_stats, _adapters, _logging_initialized

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

    for adapter_name, adapter in _adapters.items():
        try:
            adapter.cleanup()
        except Exception as e:
            logger.error(f"Error cleaning up adapter {adapter_name}: {e}")
            
    _adapters.clear()
    _session_stats.clear()
    _logging_initialized = False
    gc.collect()
    logger.info("All network resources cleaned up")


atexit.register(cleanup_resources)


def get_client(provider: tuple[str, str]):
    """Compatibility wrapper that retrieves the raw client object for third-party SDKs if needed."""
    provider_name, _ = provider
    adapter = get_adapter(provider_name)
    if hasattr(adapter, "_client"):
        return adapter._client
    return True


def chat_completion(
    messages: List[Dict[str, str]],
    temperature: float = None,
    max_tokens: int = None,
    provider: tuple[str, str] = None,
    top_p: float = None,
    debug: bool = True,
) -> Tuple[str, int, int, int, float]:
    """Generate a chat completion using the specified provider adapter."""
    
    # 1. Inherit rigid YAML defaults dynamically
    if provider is None:
        provider = (DEFAULT_PROVIDER, DEFAULT_MODEL)
    if temperature is None:
        temperature = float(DEFAULT_TEMPERATURE)
    if max_tokens is None:
        max_tokens = int(DEFAULT_MAX_TOKENS)
    if top_p is None:
        top_p = float(DEFAULT_TOP_P)

    # Ensure session and logging are initialized. This only runs once.
    _initialize_session_and_logging(debug_mode=debug)

    response_text, prompt_tokens, completion_tokens, total_tokens, latency = (
        "Error: Init failed", 0, 0, 0, 0.0,
    )
    start_time = time.monotonic()

    try:
        provider_name, model_name = provider
        # With verbose logging enabled, log the full prompt
        logger.debug(
            f"Provider: {provider}, Messages: {json.dumps(messages, indent=2)}"
        )

        adapter = get_adapter(provider_name)
        
        # Execute chat completion via the base class interface
        response_text, prompt_tokens, completion_tokens, total_tokens, latency = adapter.chat_completion(
            messages=messages,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            session=_session  # Specifically used natively by OpenRouterAdapter
        )

        # Routing adapter returned success if tokens exist! Report usage.
        if prompt_tokens > 0 or completion_tokens > 0:
            _update_stats(provider, prompt_tokens, completion_tokens)

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

    # Fallback to total elapsed time if adapter failed to track cleanly
    if latency == 0.0:
        latency = time.monotonic() - start_time

    return response_text, prompt_tokens, completion_tokens, total_tokens, latency
