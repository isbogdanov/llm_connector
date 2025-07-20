import json
from connector.connector import chat_completion


def test_llama_connection():
    """
    Test if we can connect to the llama.cpp server and get a response
    """
    # Note: This function now relies on the get_session from the connector
    from connector.connector import get_session

    session = get_session()
    # It also needs the LOCAL_LLAMA_BASE_URL, so we'll import it
    try:
        from connector_settings import LOCAL_LLAMA_BASE_URL
    except ImportError:
        import os

        LOCAL_LLAMA_BASE_URL = os.environ.get("LOCAL_LLAMA_BASE_URL")

    if not LOCAL_LLAMA_BASE_URL or "YOUR-LOCAL" in LOCAL_LLAMA_BASE_URL:
        print(
            "\nSkipping llama.cpp connection test: LOCAL_LLAMA_BASE_URL not configured."
        )
        return False

    server_url = LOCAL_LLAMA_BASE_URL

    # Test basic connectivity with a simple GET request
    try:
        # Test /health or / endpoint first
        health_url = f"{server_url.rstrip('/v1')}/health"
        print(f"Testing basic connectivity to {health_url}")
        response = session.get(health_url, timeout=5)
        print(
            f"Health endpoint response: {response.status_code} - {response.text[:100]}"
        )
        response.close()
    except Exception as e:
        print(f"Health check failed: {str(e)}")

    # Test models endpoint
    try:
        models_url = f"{server_url}/models"
        print(f"Testing models endpoint: {models_url}")
        response = session.get(models_url, timeout=5)
        print(f"Models endpoint response: {response.status_code}")
        if response.status_code == 200:
            print(f"Available models: {json.dumps(response.json(), indent=2)}")
        response.close()
    except Exception as e:
        print(f"Models endpoint check failed: {str(e)}")

    # Try a minimal chat completion
    try:
        completions_url = f"{server_url}/chat/completions"
        print(f"Testing completions endpoint: {completions_url}")

        payload = {
            "model": "anything",  # llama.cpp typically ignores this
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say hello"},
            ],
            "max_tokens": 10,
            "temperature": 0.1,
        }

        print(f"Sending payload: {json.dumps(payload, indent=2)}")
        response = session.post(
            completions_url, json=payload, timeout=30  # Longer timeout for completion
        )

        print(f"Completions response code: {response.status_code}")
        try:
            if response.status_code == 200:
                result = response.json()
                print(f"Completion successful!")
                print(f"Response: {json.dumps(result, indent=2)}")
                return True
            else:
                print(f"Error response: {response.text}")
                return False
        finally:
            response.close()

    except Exception as e:
        print(f"Completions test failed: {str(e)}")
        return False


def test_providers():
    """Test providers and return results as a dict"""
    results = {}

    # Test OpenRouter
    print("\nTesting OpenRouter:")
    try:
        openrouter_response, prompt_tokens, completion_tokens, total_tokens, latency = (
            chat_completion(
                [
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": "Hello, how are you today?"},
                ],
                provider=("openrouter", "openai/gpt-4o"),
                debug=True,
            )
        )
        results["openrouter"] = {
            "success": True,
            "response": openrouter_response,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "latency": latency,
        }
        print(f"OpenRouter response: {openrouter_response}")
    except Exception as e:
        results["openrouter"] = {"success": False, "error": str(e)}
        print(f"OpenRouter test failed: {str(e)}")

    # Test local llama
    print("\nTesting local llama.cpp:")
    try:
        local_response, prompt_tokens, completion_tokens, total_tokens, latency = (
            chat_completion(
                [
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": "Hello, how are you today?"},
                ],
                provider=("local", "llama-3-70b-versatile"),
                debug=True,
            )
        )
        results["local"] = {
            "success": True,
            "response": local_response,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "latency": latency,
        }
        print(f"Local response: {local_response}")
    except Exception as e:
        results["local"] = {"success": False, "error": str(e)}
        print(f"Local test failed: {str(e)}")

    # Test Groq API with hardcoded key
    print("\nTesting Groq:")
    try:
        groq_response, prompt_tokens, completion_tokens, total_tokens, latency = (
            chat_completion(
                [
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": "Hello, how are you today?"},
                ],
                provider=("groq", "llama-3.3-70b-versatile"),
                debug=True,
            )
        )
        results["groq"] = {
            "success": True,
            "response": groq_response,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "latency": latency,
        }
        print(f"Groq response: {groq_response}")
    except Exception as e:
        results["groq"] = {"success": False, "error": str(e)}
        print(f"Groq test failed: {str(e)}")

    return results


def main():
    """Test the connector with a simple prompt"""
    print("LLama.cpp, Groq, and OpenRouter Connector Tester")
    print("=" * 40)

    print("\n--- Testing Llama.cpp Direct Connection ---")
    test_llama_connection()
    print("\n--- Testing Provider Chat Completions ---")
    test_providers()


if __name__ == "__main__":
    main()
