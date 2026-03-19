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

import os
import yaml

from .path_resolver import CONF_DIR, CONNECTOR_HOME

try:
    from dotenv import load_dotenv
    # Explicitly load from the resolved CONNECTOR_HOME directory
    # This ensures we find .env inside the scaffolded llm-connector/ folder
    env_path = os.path.join(CONNECTOR_HOME, ".env")
    if os.path.isfile(env_path):
        load_dotenv(env_path)
    else:
        # Fallback: walk upward from CWD (legacy behavior)
        load_dotenv()
except ImportError:
    pass

def _load_yaml(filename):
    path = os.path.join(CONF_DIR, filename)
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    return {}

def _deep_merge(dict1, dict2):
    for key, value in dict2.items():
        if isinstance(value, dict) and key in dict1 and isinstance(dict1[key], dict):
            _deep_merge(dict1[key], value)
        else:
            dict1[key] = value
    return dict1

base_config = _load_yaml("llm.yaml")
override_config = _load_yaml("override.yaml")

config = _deep_merge(base_config, override_config)

# --- Expose Settings ---
DEFAULT_PROVIDER = config.get("default_provider", "openrouter")
DEFAULT_MODEL = config.get("default_model", "google/gemini-2.5-flash-lite")
DEFAULT_TEMPERATURE = config.get("default_temperature", 0.2)
DEFAULT_MAX_TOKENS = config.get("default_max_tokens", 1024)
DEFAULT_TOP_P = config.get("default_top_p", 0.7)

urls = config.get("urls", {})
LOCAL_LLAMA_BASE_URL = os.environ.get("LOCAL_LLAMA_BASE_URL", urls.get("local_llama_base_url"))
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", urls.get("ollama_base_url"))
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", urls.get("openrouter_base_url", "https://openrouter.ai/api/v1"))

# Also exposing native base URLs for future adapter overrides
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", urls.get("openai_base_url", "https://api.openai.com/v1"))
ANTHROPIC_BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", urls.get("anthropic_base_url", "https://api.anthropic.com"))
GOOGLE_BASE_URL = os.environ.get("GOOGLE_BASE_URL", urls.get("google_base_url", "https://generativelanguage.googleapis.com"))
GROQ_BASE_URL = os.environ.get("GROQ_BASE_URL", urls.get("groq_base_url", "https://api.groq.com/openai/v1"))

openrouter_headers = config.get("openrouter_headers", {})
OPENROUTER_REFERER = os.environ.get("OPENROUTER_REFERER", openrouter_headers.get("referer", ""))
OPENROUTER_SITE_NAME = os.environ.get("OPENROUTER_SITE_NAME", openrouter_headers.get("site_name", ""))