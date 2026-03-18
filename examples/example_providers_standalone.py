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

"""
Example: Using the LLM Connector to sequentially test multiple non-local providers.
Requires `.env` to be populated with your respective API keys.
"""

import sys
import os

# Dynamically add the parent directory (llm_connector root) to the python path 
# so it can seamlessly resolve `from llm_connector import chat_completion`
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_connector import chat_completion

def run_provider_test(provider_id, model_name):
    print(f"\n{'='*60}")
    print(f"Connecting to Provider: {provider_id.upper()}")
    print(f"Target Model: {model_name}")
    print(f"{'='*60}")
    
    messages = [
        {"role": "system", "content": "You are a concise, highly efficient AI system architect. Keep responses strictly under 25 words."},
        {"role": "user", "content": "Summarize the major advantage of decoupling a software architecture codebase into micro-packages."},
    ]

    print("\n--- Sent Prompt ---")
    for msg in messages:
        print(f"[{msg['role'].upper()}]: {msg['content']}")

    try:
        response, prompt_tokens, completion_tokens, total_tokens, latency = chat_completion(
            messages=messages,
            provider=(provider_id, model_name),
            temperature=0.3,
            max_tokens=256
        )

        print("\n--- Model Response ---")
        print(response.strip())
        print("\n--- Network Telemetry ---")
        print(f"Latency Time: {latency:.2f}s")
        print(f"Total Tokens: {total_tokens} (Input: {prompt_tokens}, Output: {completion_tokens})")
        
    except Exception as e:
        print(f"\n[!] Error connecting to {provider_id}: {e}")
        print("Did you ensure your API KEY is correctly configured inside the `.env` file?")

def main():
    providers_to_test = [
        ("openrouter", "anthropic/claude-3-5-haiku"),
        ("openai", "gpt-4o-mini"),
        ("anthropic", "claude-haiku-4-5"),
        ("google", "gemini-2.5-flash"),
        ("groq", "llama-3.3-70b-versatile")
    ]
    
    print("Beginning sequential multi-provider LLM Connector test...")
    
    for provider, model in providers_to_test:
        run_provider_test(provider, model)
        
    print("\n\nTest run completed! Check your `/logs` directory to view the exact USD transaction cost log automatically generated for all of these payloads.")

if __name__ == "__main__":
    main()