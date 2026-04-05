# LLM Connector

A clean, modular Python connector providing a unified interface for various Large Language Model (LLM) providers, including OpenRouter, OpenAI, Anthropic, Google (Gemini), Vertex AI, Groq, and local instances like llama.cpp and Ollama.

## Features

-   **Multiple Providers**: Connect natively to OpenRouter, Google, Vertex AI, OpenAI, Anthropic, Groq, local llama.cpp, and Ollama.
-   **YAML Configuration**: Manage all URLs, timeouts, retry logic, logging structures, and default models strictly through transparent YAML files (`llm.yaml`, `security.yaml`, `logs.yaml`).
-   **Secure Overrides**: Inject local offline IPs and enterprise pricing seamlessly via a `.gitignore` restricted `override.yaml` and keep API keys strictly in `.env`.
-   **Resilient Connections**: Features internal connection pooling, exponential backoff logic (for Rate Limits & 50x errors), and dynamically tracks OpenRouter internet pricing.
-   **Dynamic Client Loading**: Adapters load their heavy SDKs exclusively on-demand.

## Installation

### Option A: Install from PyPI (recommended)

```bash
pip install llm-connector
```

Then scaffold your project workspace:

```bash
llm-connector init
```

This creates an `llm-connector/` directory with `conf/`, `logs/`, `.env.template`, and `override.yaml.template` — everything you need to configure the connector.

## CLI Reference

The `llm-connector` command-line tool helps you set up and manage your workspace.

```bash
llm-connector --help
```

### `llm-connector init`

Scaffolds a new `llm-connector/` workspace in the current directory with all necessary configuration files.

```bash
llm-connector init [--force]
```

| Flag | Description |
|---|---|
| `--force` | Overwrite an existing `llm-connector/` directory |

**Generated structure:**

```
llm-connector/
├── .env.template              # Copy to .env and add your API keys
├── conf/
│   ├── llm.yaml               # Default provider, model, temperature, max_tokens
│   ├── security.yaml          # Connection pooling, retry limits, backoff
│   ├── logs.yaml              # Log rotation and formatting
│   └── override.yaml.template # Copy to override.yaml for local endpoints
└── logs/                      # Runtime logs (auto-created)
```

**After scaffolding:**

```bash
cd llm-connector
cp .env.template .env          # Add your API keys
cp conf/override.yaml.template conf/override.yaml  # Add local endpoints (optional)
```

### Option B: Clone as a submodule / standalone repo

```bash
git clone https://github.com/isbogdanov/llm_connector.git
cd llm_connector
pip install -e .
```

## Configuration

1.  **API Keys (`.env`)**:
    Copy the environment template and insert your secret keys.
    ```bash
    cp .env.template .env
    ```
    *API keys should exclusively live in `.env` and never be committed.*

2.  **Local Environment Overrides (`conf/override.yaml`)**:
    If you are running local inference servers (like Ollama or LLaMA.cpp), or if you want to hardcode specific contract prices, create an override file:
    ```bash
    cp conf/override.yaml.template conf/override.yaml
    ```
    *This file is safely untracked by Git, meaning your local IP addresses won't leak into the remote repository.*

3.  **Base System Configuration (`conf/llm.yaml`, `conf/logs.yaml`, `conf/security.yaml`)**:
    These files define the core tracked architecture. You can tune defaults like `default_provider`, log rotation limits, and API retry backoff-factors directly inside them.

> **Note:** When installed via pip, all paths resolve relative to your scaffolded `llm-connector/` directory. You can override this with the `LLM_CONNECTOR_HOME` environment variable.

## Usage

Import the `chat_completion` function from the connector.

```python
from llm_connector import chat_completion

# Example messages
messages = [
    {"role": "system", "content": "You are a highly capable AI assistant."},
    {"role": "user", "content": "Explain the importance of context windows in LLMs."},
]

# 1. Using the default provider & model (defined in llm.yaml)
response, p_t, c_t, t_t, latency = chat_completion(messages)
print(f"Default Response: {response}")

# 2. Specifying a specific provider and native model
response_openrouter, _, _, _, _ = chat_completion(
    messages,
    provider=("openrouter", "anthropic/claude-3.5-sonnet"),
    temperature=0.5,
)

# 3. Using Vertex AI
response_vertex, _, _, _, _ = chat_completion(
    messages,
    provider=("vertex", "gemini-2.5-flash"),
)

# 4. Hitting a local offline model
response_local, _, _, _, _ = chat_completion(
    messages,
    provider=("ollama", "llama3.1:8b"),
)
```

## Creating Custom Adapters

This package is designed to be infinitely extensible. To add a brand-new API provider, follow these 3 steps:

1. **Create the Adapter Class**:
   Add a new file in `llm_connector/adapters/` (e.g., `my_adapter.py`). It must inherit from `AdapterBase` and strictly implement the core `chat_completion` signature:
   ```python
   from .adapter import AdapterBase

   class MyCustomAdapter(AdapterBase):
       def chat_completion(self, messages, model, temperature, max_tokens, top_p, **kwargs):
           # 1. Initialize your specific SDK client here
           # 2. Translate the generic 'messages' array into your provider's requested format
           # 3. Await the response
           # 4. Extract token usage and latency
           
           return response_text, prompt_tokens, completion_tokens, total_tokens, latency
   ```

2. **Export the Adapter**:
   Expose your new adapter class inside `llm_connector/adapters/__init__.py`:
   ```python
   from .my_adapter import MyCustomAdapter
   __all__ = [..., "MyCustomAdapter"]
   ```

3. **Register it in the Router**:
   Add your provider string internally into the `get_adapter` network factory inside `llm_connector/connector.py`:
   ```python
   elif provider_name == "my-custom-api":
       _adapters[provider_name] = MyCustomAdapter()
   ```

## Testing

The testing suite natively utilizes `pytest` to rigidly guarantee the dynamic YAML hierarchy maps flawlessly to the internal routing logic without triggering infinite fallback loops.

To execute the entire engine diagnostic comprehensively, run:
```bash
pytest tests/
```

To include local model tests (requires a running llama.cpp or Ollama server):
```bash
pytest tests/ --run-local
```

The local suite actively asserts:
1. **Integration Verification**: Dynamically attempts to route offline dummy prompts safely through OpenRouter, OpenAI, Anthropic, Google, Groq, and your local/Ollama networking blocks.
2. **Security Validation**: Explicitly deconstructs the `requests.Session()` engine upon boot and validates your exact `security.yaml` limits (Connection Pools, HTTP Max Retries, and network `backoff_factors`) are clamped physically to the memory pipeline.
3. **Adapter Isolation**: Prevents SDK crosstalk by verifying requests generated for `"groq"` cannot accidentally bleed over or trigger `"local"` internal logic modules.