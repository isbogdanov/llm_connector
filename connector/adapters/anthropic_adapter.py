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
from typing import List, Dict, Tuple
from .adapter import AdapterBase

try:
    import anthropic
    ANTHROPIC_SDK_AVAILABLE = True
except ImportError:
    ANTHROPIC_SDK_AVAILABLE = False
    anthropic = None

import os
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

logger = logging.getLogger("LLMConnector")


class AnthropicAdapter(AdapterBase):
    def __init__(self):
        self._client = None
        
    def _initialize_client(self):
        if self._client is None:
            if not ANTHROPIC_SDK_AVAILABLE:
                raise ValueError("anthropic package is not installed. Install it with: pip install anthropic")
            if not ANTHROPIC_API_KEY:
                raise ValueError("ANTHROPIC_API_KEY is not set.")
            self._client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            logger.info("Created new native Anthropic client")
            
    def _convert_messages_to_anthropic_format(self, messages):
        anthropic_messages = []
        system_instruction = ""
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                system_instruction += content + "\n"
            else:
                anthropic_messages.append({"role": role, "content": content})
        
        return anthropic_messages, system_instruction.strip()

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
        logger.info(f"Requesting completion from anthropic with model {model}")
        
        response_text, prompt_tokens, completion_tokens, total_tokens, latency = (
            None, 0, 0, 0, 0.0,
        )
        start_call_time = time.monotonic()
        
        anthropic_messages, system_instruction = self._convert_messages_to_anthropic_format(messages)
        
        try:
            kwargs_dict = {
                "model": model,
                "messages": anthropic_messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if system_instruction:
                kwargs_dict["system"] = system_instruction

            response = self._client.messages.create(**kwargs_dict)
            latency = time.monotonic() - start_call_time

            if response and response.content:
                # Anthropic groups response text in blocks
                response_text = response.content[0].text
                logger.debug(f"LLM Response: {response_text}")
                if hasattr(response, 'usage'):
                    prompt_tokens = getattr(response.usage, "input_tokens", 0) or 0
                    completion_tokens = getattr(response.usage, "output_tokens", 0) or 0
                    total_tokens = prompt_tokens + completion_tokens
            else:
                response_text = "Error: No response/choices received from API"
                
        except Exception as exception:
            response_text = f"Error with Anthropic API: {type(exception).__name__}: {exception}"
            logger.error(response_text)
            
        return response_text, prompt_tokens, completion_tokens, total_tokens, latency
        
    def cleanup(self) -> None:
        if self._client:
            self._client.close()
            self._client = None