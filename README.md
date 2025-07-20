# LLM Connector

A simple Python connector with unified interface for various Large Language Model (LLM) providers, including OpenRouter, Groq, and local instances like llama.cpp and Ollama.

## Features

-   **Multiple Providers**: Connect to OpenRouter, Groq, local llama.cpp, and Ollama.
-   **Centralized Configuration**: Manage all API keys, URLs, and settings in a single `connector_settings.py` file.
-   **Resilient Connections**: Uses persistent sessions with connection pooling and automatic retries.
-   **Dynamic Client Loading**: Clients for different providers are loaded on-demand.
-   **Clear Logging**: Provides detailed logs for requests and responses.

## Installation

1.  **Clone the repository as a submodule in your project:**

    The top-level directory is `llm_connector`, and the Python package is in the `connector/` subdirectory.

    ```bash
    git submodule add <repository_url> llm_connector
    git submodule update --init --recursive
    ```
    *(You can replace `llm_connector` with a different path if you prefer).*

2.  **Install the required dependencies:**

    ```bash
    pip install -r llm_connector/requirements.txt
    ```

## Configuration

1.  **Create your settings file:**

    Copy the template to a new file named `connector_settings.py`. This file should be placed in a location that is part of your `PYTHONPATH` (like your project's root) so the connector can import it.

    ```bash
    cp llm_connector/connector/connector_settings.py.template path/to/your_project/connector_settings.py
    ```

    **Important**: The `connector_settings.py` file is included in the `.gitignore` to prevent committing sensitive information.

2.  **Set your API keys and endpoints:**

    Open `connector_settings.py` and fill in your API keys and base URLs. It is highly recommended to use environment variables for this.

    ```bash
    # Example for setting environment variables
    export OPENROUTER_API_KEY="your_openrouter_key"
    export GROQ_API_KEY="your_groq_key"
    export LOCAL_LLAMA_BASE_URL="http://localhost:8080/v1"
    ```

## Usage

Import the `chat_completion` function from the connector. For the import to work, the `llm_connector` directory must be in your Python path.

```python
import sys
# Add the submodule's path to the Python path
sys.path.append('llm_connector')

from connector.connector import chat_completion

# Example messages
messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "Explain the importance of context windows in LLMs."},
]

# Using the default provider from settings
response, p_tokens, c_tokens, t_tokens, latency = chat_completion(messages)
print(f"Response: {response}")
print(f"Latency: {latency:.2f}s")

# Specifying a different provider and model
response_groq, _, _, _, _ = chat_completion(
    messages,
    provider=("groq", "llama-3.1-70b-versatile"),
    temperature=0.5,
)
print(f"Groq Response: {response_groq}")
```

## Testing

The connector includes a testing utility to verify your connections to the different providers.

To run the tests, execute the `test_connector.py` script from the root of the `llm_connector` repository:

```bash
python test_connector.py
``` 