# Copyright 2026 Igor Bogdanov
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

import pytest
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import importlib
try:
    settings = importlib.import_module("llm_connector.helpers.llm_settings")
    DEFAULT_MODEL = getattr(settings, "DEFAULT_MODEL", "google/gemini-2.5-flash")
except ImportError:
    DEFAULT_MODEL = "google/gemini-2.5-flash"


def are_credentials_present(provider):
    if provider == "openrouter":
        val = os.environ.get("OPENROUTER_API_KEY")
        return bool(val and "YOUR-" not in val)
    if provider == "groq":
        val = os.environ.get("GROQ_API_KEY")
        return bool(val and "YOUR-" not in val)
    if provider == "local":
        from llm_connector.helpers.llm_settings import LOCAL_LLAMA_BASE_URL
        return bool(LOCAL_LLAMA_BASE_URL)
    if provider == "ollama":
        from llm_connector.helpers.llm_settings import OLLAMA_BASE_URL
        return bool(OLLAMA_BASE_URL)
    if provider == "google":
        val = os.environ.get("GOOGLE_API_KEY")
        return bool(val and "YOUR-" not in val)
    if provider == "openai":
        val = os.environ.get("OPENAI_API_KEY")
        return bool(val and "YOUR-" not in val)
    if provider == "anthropic":
        val = os.environ.get("ANTHROPIC_API_KEY")
        return bool(val and "YOUR-" not in val)
    return False

# --- Integration and Acceptance Tests ---

@pytest.fixture
def test_messages():
    """Provides a standard set of messages for acceptance tests."""
    return [
        {
            "role": "system",
            "content": "You are a helpful AI assistant. Respond in one sentence.",
        },
        {"role": "user", "content": "What is the capital of France?"},
    ]

@pytest.mark.skipif(not are_credentials_present("openrouter"), reason="OpenRouter API key not configured.")
def test_openrouter_acceptance(test_messages):
    from llm_connector import chat_completion
    model = DEFAULT_MODEL
    response, _, _, _, latency = chat_completion(test_messages, provider=("openrouter", model))
    print(f"\nOpenRouter ({model}) Response (Latency: {latency:.2f}s): {response}")
    assert "Paris" in response

@pytest.mark.skipif(not are_credentials_present("groq"), reason="Groq API key not configured.")
def test_groq_acceptance(test_messages):
    from llm_connector import chat_completion
    model = "llama-3.1-8b-instant"
    response, _, _, _, latency = chat_completion(test_messages, provider=("groq", model))
    print(f"\nGroq ({model}) Response (Latency: {latency:.2f}s): {response}")
    assert "Paris" in response

@pytest.mark.skipif(not are_credentials_present("google"), reason="Google API key not configured.")
def test_google_acceptance(test_messages):
    from llm_connector import chat_completion
    model = "gemini-2.5-flash"
    response, _, _, _, latency = chat_completion(test_messages, provider=("google", model))
    print(f"\nGoogle ({model}) Response (Latency: {latency:.2f}s): {response}")
    assert "Paris" in response

@pytest.mark.skipif(not are_credentials_present("openai"), reason="OpenAI API key not configured.")
def test_openai_acceptance(test_messages):
    from llm_connector import chat_completion
    model = "gpt-4o-mini"
    response, _, _, _, latency = chat_completion(test_messages, provider=("openai", model))
    print(f"\nOpenAI ({model}) Response (Latency: {latency:.2f}s): {response}")
    assert "Paris" in response

@pytest.mark.skipif(not are_credentials_present("anthropic"), reason="Anthropic API key not configured.")
def test_anthropic_acceptance(test_messages):
    from llm_connector import chat_completion
    model = "claude-haiku-4-5"
    response, _, _, _, latency = chat_completion(test_messages, provider=("anthropic", model))
    print(f"\nAnthropic ({model}) Response (Latency: {latency:.2f}s): {response}")
    assert "Paris" in response

@pytest.mark.local_model
@pytest.mark.skipif(not are_credentials_present("local"), reason="Local server not configured.")
def test_local_llama_acceptance(test_messages):
    from llm_connector import chat_completion
    response, _, _, _, latency = chat_completion(test_messages, provider=("local", "local-model"))
    print(f"\nLocal Llama Response (Latency: {latency:.2f}s): {response}")
    assert "Error" not in response

@pytest.mark.local_model
@pytest.mark.skipif(not are_credentials_present("ollama"), reason="Ollama server not configured.")
def test_ollama_acceptance(test_messages):
    from llm_connector import chat_completion
    model = "llama3.1:8b"
    response, _, _, _, latency = chat_completion(test_messages, provider=("ollama", model))
    print(f"\nOllama ({model}) Response (Latency: {latency:.2f}s): {response}")
    assert "Error" not in response