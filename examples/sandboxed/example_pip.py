"""
Standalone example: using llm-connector installed via pip.

Prerequisites:
    pip install llm-connector
    llm-connector init
    # Then edit llm-connector/.env with your API keys
"""
# Copyright 2026 Igor Bogdanov
# Licensed under the Apache License, Version 2.0

from llm_connector import chat_completion, cleanup_resources

def main():
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant. Respond in one sentence."},
        {"role": "user", "content": "What is the capital of France?"},
    ]

    # --- 1. Default provider (from llm.yaml) ---
    print("=== Default Provider ===")
    response, prompt_t, comp_t, total_t, latency = chat_completion(messages)
    print(f"Response: {response}")
    print(f"Tokens: {prompt_t} prompt, {comp_t} completion | Latency: {latency:.2f}s")

    # --- 2. Explicit provider routing ---
    providers = [
        ("openrouter", "google/gemini-2.5-flash-lite"),
        ("google", "gemini-2.5-flash"),
        ("openai", "gpt-4o-mini"),
        ("anthropic", "claude-haiku-4-5"),
        ("groq", "llama-3.1-8b-instant"),
        ("vertex", "gemini-2.5-flash"),
    ]

    for provider_name, model in providers:
        print(f"\n=== {provider_name} / {model} ===")
        try:
            response, p, c, t, latency = chat_completion(
                messages, provider=(provider_name, model)
            )
            print(f"Response: {response}")
            print(f"Tokens: {p} prompt, {c} completion | Latency: {latency:.2f}s")
        except Exception as e:
            print(f"Skipped ({e})")

    # --- 3. Clean up and print session summary ---
    cleanup_resources()


if __name__ == "__main__":
    main()
