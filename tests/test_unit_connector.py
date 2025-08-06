import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture(autouse=True)
def reset_connector_globals():
    """
    This fixture automatically runs before each test in this file.
    It resets the global client caches in the connector module to ensure
    that tests run in a clean, isolated state.
    """
    import connector.connector

    connector.connector._groq_client = None
    connector.connector._openai_clients = {}
    # We also reset the session to be thorough
    if connector.connector._session:
        connector.connector._session.close()
    connector.connector._session = None


# --- Unit Tests ---


def test_get_client():
    """Test the get_client function logic in isolation."""
    # Import here to ensure we get a fresh module state
    import connector.connector

    # Manually reset state to guarantee isolation for this test
    connector.connector._groq_client = None
    connector.connector._openai_clients = {}

    # Test Groq client creation
    with patch("connector.connector.Groq") as mock_groq:
        client = connector.connector.get_client(("groq", "test-model"))
        mock_groq.assert_called_once()
        assert client is not None

    # Test local client creation
    with patch("connector.connector.openai.OpenAI") as mock_openai:
        client = connector.connector.get_client(("local", "test-model"))
        mock_openai.assert_called_once()
        assert client is not None

    # Test that clients are cached
    with patch("connector.connector.Groq") as mock_groq_cached:
        # The client should be returned from cache, so the mock should NOT be called
        client = connector.connector.get_client(("groq", "test-model"))
        mock_groq_cached.assert_not_called()
        assert client is not None

    # Test OpenRouter (should return None)
    client = connector.connector.get_client(("openrouter", "test-model"))
    assert client is None

    # Test unsupported provider
    with pytest.raises(ValueError, match="Unsupported provider"):
        connector.connector.get_client(("unsupported", "test-model"))


@patch("connector.connector.get_session")
def test_openrouter_chat_completion_success(mock_get_session):
    """Test a successful OpenRouter API call using mocks."""
    from connector.connector import openrouter_chat_completion

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Test response"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }
    mock_get_session.return_value.post.return_value = mock_response

    response, p_tokens, c_tokens, t_tokens, latency = openrouter_chat_completion(
        messages=[{"role": "user", "content": "Hello"}],
        model="test-model",
        temperature=0.5,
        max_tokens=50,
        top_p=0.7,
    )
    assert response == "Test response"
    assert p_tokens == 10
    assert c_tokens == 20
    assert t_tokens == 30
    assert latency > 0


@patch("connector.connector.openrouter_chat_completion")
@patch("connector.connector.get_client")
def test_chat_completion_routing(mock_get_client, mock_openrouter_call):
    """Test that chat_completion routes to the correct function based on provider."""
    from connector.connector import chat_completion

    chat_completion(messages=[], provider=("openrouter", "test-model"))
    mock_openrouter_call.assert_called_once()
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Groq response"))]
    mock_response.usage = MagicMock(
        prompt_tokens=5, completion_tokens=10, total_tokens=15
    )
    mock_client.chat.completions.create.return_value = mock_response
    chat_completion(messages=[], provider=("groq", "test-model"))
    mock_get_client.assert_called_with(("groq", "test-model"))
    mock_client.chat.completions.create.assert_called_once()
