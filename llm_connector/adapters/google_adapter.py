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
    from google import genai
    from google.genai import types
    GOOGLE_SDK_AVAILABLE = True
except ImportError:
    GOOGLE_SDK_AVAILABLE = False
    genai = None
    types = None

import os
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

logger = logging.getLogger("LLMConnector")


class GoogleAdapter(AdapterBase):
    def __init__(self):
        self._client = None

    def _initialize_client(self):
        if self._client is None:
            if not GOOGLE_SDK_AVAILABLE:
                raise ValueError("google-genai package is not installed. Install it with: pip install google-genai")
            if not GOOGLE_API_KEY:
                raise ValueError("GOOGLE_API_KEY is not set.")
            self._client = genai.Client(api_key=GOOGLE_API_KEY)
            logger.info("Created new Google GenAI client")

    def _convert_messages_to_google_format(self, messages):
        """Convert OpenAI-style messages to Google GenAI contents format."""
        contents = []
        system_instruction = None
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                system_instruction = content
            elif role == "user":
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=content)]
                ))
            elif role == "assistant":
                contents.append(types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=content)]
                ))
        
        return contents, system_instruction

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
        
        response_text, prompt_tokens, completion_tokens, total_tokens, latency = (
            None, 0, 0, 0, 0.0,
        )
        
        max_retries = 5
        base_delay = 1.0
        max_delay = 60.0
        
        start_time = time.monotonic()
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                contents, system_instruction = self._convert_messages_to_google_format(messages)
                
                config = types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    top_p=top_p,
                )
                if system_instruction:
                    config.system_instruction = system_instruction
                
                logger.info(f"Requesting completion from Google GenAI with model {model} (attempt {attempt + 1}/{max_retries + 1})")
                
                response = self._client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                )
                
                latency = time.monotonic() - start_time
                
                if response and response.text:
                    response_text = response.text
                    logger.debug(f"LLM Response: {response_text}")
                    
                    if hasattr(response, 'usage_metadata') and response.usage_metadata:
                        prompt_tokens = response.usage_metadata.prompt_token_count or 0
                        completion_tokens = response.usage_metadata.candidates_token_count or 0
                        total_tokens = response.usage_metadata.total_token_count or 0
                    else:
                        logger.warning("No usage metadata available from Google API response")
                    
                    return response_text, prompt_tokens, completion_tokens, total_tokens, latency
                else:
                    response_text = "Error: No response text received from Google API"
                    logger.error(response_text)
                    return response_text, prompt_tokens, completion_tokens, total_tokens, latency
                    
            except Exception as exception:
                last_exception = exception
                error_str = str(exception).lower()
                
                is_rate_limit = (
                    "429" in str(exception) or 
                    "resource exhausted" in error_str or
                    "quota" in error_str or
                    "rate limit" in error_str or
                    "too many requests" in error_str
                )
                
                if is_rate_limit and attempt < max_retries:
                    import random
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    jitter = delay * 0.25 * (2 * random.random() - 1)
                    actual_delay = delay + jitter
                    
                    logger.warning(
                        f"Rate limit hit (attempt {attempt + 1}/{max_retries + 1}). "
                        f"Retrying in {actual_delay:.2f}s (base: {delay:.1f}s, jitter: {jitter:.2f}s)"
                    )
                    time.sleep(actual_delay)
                    continue
                else:
                    response_text = f"Error with Google API: {type(exception).__name__}: {exception}"
                    logger.error(response_text)
                    if attempt >= max_retries and is_rate_limit:
                        logger.error(f"Max retries ({max_retries}) exceeded for rate limiting. Giving up.")
                    break
        
        if latency == 0.0:
            latency = time.monotonic() - start_time
        
        if response_text is None:
            response_text = f"Error with Google API after {max_retries} retries: {type(last_exception).__name__}: {last_exception}"
        
        return response_text, prompt_tokens, completion_tokens, total_tokens, latency