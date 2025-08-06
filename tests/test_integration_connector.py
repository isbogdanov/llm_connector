import pytest
import importlib

# --- Pytest Configuration for Integration Tests ---

try:
    settings = importlib.import_module("connector.connector_settings")
    SETTINGS_PRESENT = True
except ImportError:
    settings = None
    SETTINGS_PRESENT = False


def are_credentials_present(provider):
    if not SETTINGS_PRESENT:
        return False
    if provider == "openrouter":
        return bool(
            settings.OPENROUTER_API_KEY and "sk-or-v1" in settings.OPENROUTER_API_KEY
        )
    if provider == "groq":
        return bool(settings.GROQ_API_KEY and "gsk_" in settings.GROQ_API_KEY)
    if provider == "local":
        return bool(
            settings.LOCAL_LLAMA_BASE_URL
            and "YOUR-LOCAL" not in settings.LOCAL_LLAMA_BASE_URL
        )
    if provider == "ollama":
        return bool(
            settings.OLLAMA_BASE_URL and "YOUR-OLLAMA" not in settings.OLLAMA_BASE_URL
        )
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


@pytest.mark.skipif(
    not are_credentials_present("openrouter"),
    reason="OpenRouter API key not configured.",
)
def test_openrouter_acceptance(test_messages):
    from connector.connector import chat_completion

    model = settings.DEFAULT_MODEL
    response, _, _, _, latency = chat_completion(
        test_messages, provider=("openrouter", model)
    )
    print(f"\nOpenRouter ({model}) Response (Latency: {latency:.2f}s): {response}")
    assert "Paris" in response


@pytest.mark.skipif(
    not are_credentials_present("groq"), reason="Groq API key not configured."
)
def test_groq_acceptance(test_messages):
    from connector.connector import chat_completion

    model = "llama-3.1-8b-instant"
    response, _, _, _, latency = chat_completion(
        test_messages, provider=("groq", model)
    )
    print(f"\nGroq ({model}) Response (Latency: {latency:.2f}s): {response}")
    assert "Paris" in response


@pytest.mark.skipif(
    not are_credentials_present("local"), reason="Local Llama server not configured."
)
def test_local_llama_acceptance(test_messages):
    from connector.connector import chat_completion

    response, _, _, _, latency = chat_completion(
        test_messages, provider=("local", "local-model")
    )
    print(f"\nLocal Llama Response (Latency: {latency:.2f}s): {response}")
    assert response is not None
    assert "Error" not in response


@pytest.mark.skipif(
    not are_credentials_present("ollama"), reason="Ollama server not configured."
)
def test_ollama_acceptance(test_messages):
    from connector.connector import chat_completion

    model = "llama3.1:8b"
    response, _, _, _, latency = chat_completion(
        test_messages, provider=("ollama", model)
    )
    print(f"\nOllama ({model}) Response (Latency: {latency:.2f}s): {response}")
    assert response is not None
    assert "Error" not in response
