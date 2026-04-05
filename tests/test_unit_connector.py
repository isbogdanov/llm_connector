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
from unittest.mock import patch, MagicMock

@pytest.fixture(autouse=True)
def reset_connector_globals():
    """
    Reset global adapter caches to ensure tests run in an isolated state.
    """
    import llm_connector.connector
    llm_connector.connector.cleanup_resources()

# --- Unit Tests ---

def test_get_adapter():
    """Test that the adapter factory returns the correct classes based on provider."""
    print("\n[TEST] Verifying the factory accurately resolves all supported API Adapters.")
    import llm_connector.connector
    from llm_connector.adapters import GroqAdapter, LocalAdapter, OpenRouterAdapter, GoogleAdapter, OpenAIAdapter, AnthropicAdapter, VertexAdapter

    llm_connector.connector.cleanup_resources()

    # Test that each provider correctly spins up its adapter
    assert isinstance(llm_connector.connector.get_adapter("groq"), GroqAdapter)
    assert isinstance(llm_connector.connector.get_adapter("local"), LocalAdapter)
    assert isinstance(llm_connector.connector.get_adapter("openrouter"), OpenRouterAdapter)
    assert isinstance(llm_connector.connector.get_adapter("google"), GoogleAdapter)
    assert isinstance(llm_connector.connector.get_adapter("openai"), OpenAIAdapter)
    assert isinstance(llm_connector.connector.get_adapter("anthropic"), AnthropicAdapter)
    assert isinstance(llm_connector.connector.get_adapter("vertex"), VertexAdapter)

    # Test unsupported provider
    with pytest.raises(ValueError, match="Unsupported provider"):
        llm_connector.connector.get_adapter("unsupported")

@patch("llm_connector.adapters.openrouter_adapter.OpenRouterAdapter.chat_completion")
def test_openrouter_chat_completion_success(mock_openrouter_call):
    """Test that the chat_completion router hits the correct adapter method."""
    print("\n[TEST] Verifying dynamic token and latency passthrough routing natively.")
    from llm_connector import chat_completion

    mock_openrouter_call.return_value = ("Test response", 10, 20, 30, 1.5)

    response, p_tokens, c_tokens, t_tokens, latency = chat_completion(
        messages=[{"role": "user", "content": "Hello"}],
        provider=("openrouter", "test-model"),
        temperature=0.5,
        max_tokens=50,
        top_p=0.7,
    )
    
    mock_openrouter_call.assert_called_once()
    assert response == "Test response"
    assert p_tokens == 10
    assert c_tokens == 20
    assert t_tokens == 30
    assert latency > 0

@patch("llm_connector.adapters.groq_adapter.GroqAdapter.chat_completion")
@patch("llm_connector.adapters.local_adapter.LocalAdapter.chat_completion")
def test_chat_completion_routing(mock_local_call, mock_groq_call):
    """Test that chat_completion routes to the correct function based on provider string."""
    print("\n[TEST] Ensuring explicit endpoint routing executes strictly without crosstalk.")
    from llm_connector import chat_completion

    mock_groq_call.return_value = ("Groq response", 5, 10, 15, 0.5)
    mock_local_call.return_value = ("Local response", 2, 4, 6, 0.1)

    chat_completion(messages=[], provider=("groq", "test-model"))
    mock_groq_call.assert_called_once()

    chat_completion(messages=[], provider=("local", "test-model"))
    mock_local_call.assert_called_once()

def test_security_session_configuration():
    """Test that the internal requests.Session is configured with the correct security parameters."""
    print("\n[TEST] Validating strict security.yaml parameters map securely to the urllib3 pooling limits.")
    import llm_connector.connector
    from llm_connector.helpers.security_settings import RETRY_MAX, RETRY_BACKOFF, POOL_CONNECTIONS, POOL_MAXSIZE
    
    # Ensure fresh state
    llm_connector.connector._logging_initialized = False
    llm_connector.connector._initialize_session_and_logging()
    
    session = llm_connector.connector.get_session()
    
    # Check that http and https adapters are mounted and configured
    http_adapter = session.adapters.get("http://")
    https_adapter = session.adapters.get("https://")
    
    assert http_adapter is not None
    assert https_adapter is not None
    
    # Validate Pool sizes
    assert http_adapter._pool_connections == POOL_CONNECTIONS
    assert http_adapter._pool_maxsize == POOL_MAXSIZE
    assert https_adapter._pool_connections == POOL_CONNECTIONS
    assert https_adapter._pool_maxsize == POOL_MAXSIZE
    
    # Validate Retry Backoff parameters
    retry_strategy = http_adapter.max_retries
    assert retry_strategy.total == RETRY_MAX
    assert retry_strategy.backoff_factor == RETRY_BACKOFF
    assert 429 in retry_strategy.status_forcelist
    assert 500 in retry_strategy.status_forcelist