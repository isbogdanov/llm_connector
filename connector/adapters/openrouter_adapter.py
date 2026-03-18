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
import requests
import logging
from typing import List, Dict, Tuple
from .adapter import AdapterBase
import os
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

try:
    from ..helpers.llm_settings import OPENROUTER_REFERER, OPENROUTER_SITE_NAME, OPENROUTER_BASE_URL
except ImportError:
    OPENROUTER_REFERER = os.environ.get("OPENROUTER_REFERER", "")
    OPENROUTER_SITE_NAME = os.environ.get("OPENROUTER_SITE_NAME", "")
    OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

logger = logging.getLogger("LLMConnector")

class OpenRouterAdapter(AdapterBase):
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        **kwargs
    ) -> Tuple[str, int, int, int, float]:
        session = kwargs.get("session")
        if not session:
            raise ValueError("OpenRouterAdapter requires a 'session' object passed in kwargs.")
            
        if not OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY is not set.")
            
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": OPENROUTER_REFERER,
            "X-Title": OPENROUTER_SITE_NAME,
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }
        request_url = f"{OPENROUTER_BASE_URL}/chat/completions"

        response_text, prompt_tokens, completion_tokens, total_tokens, latency = (
            None, 0, 0, 0, 0.0,
        )
        start_time = time.monotonic()

        try:
            logger.info(f"HTTP Request: POST {request_url} for model {model}")
            response = session.post(
                request_url, headers=headers, json=payload, timeout=(3.05, 60)
            )
            logger.info(f"HTTP Response: {response.status_code} {response.reason}")
            response.raise_for_status()
            
            response_data = response.json()
            if "choices" in response_data and response_data["choices"]:
                response_text = response_data["choices"][0]["message"]["content"]
                logger.debug(f"LLM Response: {response_text}")
                usage = response_data.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                total_tokens = usage.get("total_tokens", 0)
            else:
                response_text = "Error: Unexpected response format from OpenRouter"
                logger.error(f"{response_text}: {response_data}")
                
        except requests.exceptions.RequestException as exception:
            response_text = f"Error with OpenRouter API: {type(exception).__name__}: {exception}"
            logger.error(response_text)
            if hasattr(exception, "response") and exception.response:
                logger.error(f"Response status: {exception.response.status_code}")
                logger.error(f"Response text: {exception.response.text}")
                
        finally:
            latency = time.monotonic() - start_time
            if "response" in locals() and response:
                try:
                    response.close()
                except:
                    pass

        return response_text, prompt_tokens, completion_tokens, total_tokens, latency