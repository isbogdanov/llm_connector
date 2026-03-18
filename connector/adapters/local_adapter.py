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

import time
import logging
import openai
from typing import List, Dict, Tuple
from .adapter import AdapterBase
import os
try:
    from ..helpers.llm_settings import LOCAL_LLAMA_BASE_URL, OLLAMA_BASE_URL
except ImportError:
    LOCAL_LLAMA_BASE_URL = os.environ.get("LOCAL_LLAMA_BASE_URL")
    OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL")

logger = logging.getLogger("LLMConnector")

LOCAL_LLAMA_CONFIG = {"api_key": "not-needed", "base_url": LOCAL_LLAMA_BASE_URL}
OLLAMA_CONFIG = {"api_key": "ollama", "base_url": OLLAMA_BASE_URL}

class LocalAdapter(AdapterBase):
    def __init__(self, provider_name: str):
        if provider_name not in ["local", "ollama"]:
            raise ValueError(f"Unsupported local provider: {provider_name}")
        self.provider_name = provider_name
        self._client = None
        
    def _initialize_client(self):
        if self._client is None:
            config = LOCAL_LLAMA_CONFIG if self.provider_name == "local" else OLLAMA_CONFIG
            self._client = openai.OpenAI(**config)
            logger.info(f"Created new {self.provider_name} local client")
            
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        **kwargs
    ) -> Tuple[str, int, int, int, float]:
        self._initialize_client()
        logger.info(f"Requesting completion from {self.provider_name} with model {model}")
        
        response_text, prompt_tokens, completion_tokens, total_tokens, latency = (
            None, 0, 0, 0, 0.0,
        )
        start_call_time = time.monotonic()
        
        response = self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        latency = time.monotonic() - start_call_time

        if response and response.choices:
            response_text = response.choices[0].message.content
            logger.debug(f"LLM Response: {response_text}")
            if response.usage:
                prompt_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
                completion_tokens = getattr(response.usage, "completion_tokens", 0) or 0
                total_tokens = getattr(response.usage, "total_tokens", 0) or 0
        else:
            response_text = "Error: No response/choices received from API"
            
        return response_text, prompt_tokens, completion_tokens, total_tokens, latency
        
    def cleanup(self) -> None:
        self._client = None