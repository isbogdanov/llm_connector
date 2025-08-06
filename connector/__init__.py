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

# Make the main functions and classes available for easier import
from .connector import (
    chat_completion,
    get_client,
    get_session,
    cleanup_resources,
)
from .connector_settings import (
    OPENROUTER_API_KEY,
    GROQ_API_KEY,
    LOCAL_LLAMA_BASE_URL,
    OLLAMA_BASE_URL,
)

__all__ = [
    "chat_completion",
    "get_client",
    "get_session",
    "cleanup_resources",
    "OPENROUTER_API_KEY",
    "GROQ_API_KEY",
    "LOCAL_LLAMA_BASE_URL",
    "OLLAMA_BASE_URL",
]
