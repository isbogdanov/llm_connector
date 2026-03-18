# Sandboxed Pip Example

This example demonstrates using `llm-connector` as a pip-installed package in a clean virtual environment — completely isolated from the source repo.

## Setup

```bash
# 1. Create a fresh virtual environment
python -m venv .venv
source .venv/bin/activate

# 2. Install from PyPI
pip install llm-connector

# 3. Scaffold the workspace
llm-connector init

# 4. Configure your API keys
cp llm-connector/.env.template llm-connector/.env
# Edit llm-connector/.env and add your keys

# 5. Run the example
python example_pip.py
```
