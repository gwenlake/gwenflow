from typing import TypedDict


class ModelInfo(TypedDict):
    context_window: int
    reasoning: bool


# Last verified: 2026-07-21
MODELS: dict[str, ModelInfo] = {
    # --- OPENAI ---
    "gpt-5.6": {"context_window": 1050000, "reasoning": True},  # alias -> gpt-5.6-sol
    "gpt-5.6-sol": {"context_window": 1050000, "reasoning": True},
    "gpt-5.6-terra": {"context_window": 1050000, "reasoning": True},
    "gpt-5.6-luna": {"context_window": 1050000, "reasoning": True},
    "gpt-5.5-pro": {"context_window": 1050000, "reasoning": True},
    "gpt-5.5": {"context_window": 1050000, "reasoning": True},
    "gpt-5.4-pro": {"context_window": 1050000, "reasoning": True},
    "gpt-5.4": {"context_window": 1050000, "reasoning": True},
    "gpt-5.4-mini": {"context_window": 400000, "reasoning": True},
    "gpt-5.4-nano": {"context_window": 400000, "reasoning": True},
    "gpt-5.2-pro": {"context_window": 400000, "reasoning": True},
    "gpt-5.2": {"context_window": 400000, "reasoning": True},
    "gpt-5": {"context_window": 400000, "reasoning": True},
    "gpt-5-mini": {"context_window": 400000, "reasoning": True},
    "gpt-5-nano": {"context_window": 400000, "reasoning": True},
    "gpt-4.5": {"context_window": 128000, "reasoning": False},  # retired from API 2025-07-14
    "gpt-4.1": {"context_window": 1047576, "reasoning": False},
    "gpt-4.1-mini": {"context_window": 1047576, "reasoning": False},
    "gpt-4.1-nano": {"context_window": 1047576, "reasoning": False},
    "gpt-4o": {"context_window": 128000, "reasoning": False},
    "gpt-4o-mini": {"context_window": 128000, "reasoning": False},
    "o1": {"context_window": 200000, "reasoning": True},
    "o1-preview": {"context_window": 128000, "reasoning": True},  # retired from API 2025-07-28
    "o1-mini": {"context_window": 128000, "reasoning": True},  # retired from API 2025-10
    "o3": {"context_window": 200000, "reasoning": True},
    "o3-pro": {"context_window": 200000, "reasoning": True},
    "o3-mini": {"context_window": 200000, "reasoning": True},
    # not an API model id: use o3-mini with reasoning_effort="high"
    "o3-mini-high": {"context_window": 200000, "reasoning": True},
    "o4-mini": {"context_window": 200000, "reasoning": True},
    # --- ANTHROPIC ---
    "claude-fable-5": {"context_window": 1000000, "reasoning": True},
    "claude-mythos-5": {"context_window": 1000000, "reasoning": True},
    "claude-opus-4-8": {"context_window": 1000000, "reasoning": True},
    "claude-opus-4-7": {"context_window": 1000000, "reasoning": True},
    "claude-opus-4-6": {"context_window": 1000000, "reasoning": True},
    "claude-opus-4-5": {"context_window": 200000, "reasoning": True},
    "claude-opus-4-1": {"context_window": 200000, "reasoning": True},
    "claude-sonnet-5": {"context_window": 1000000, "reasoning": True},
    "claude-sonnet-4-6": {"context_window": 1000000, "reasoning": True},
    "claude-sonnet-4-5": {"context_window": 200000, "reasoning": False},
    "claude-haiku-4-5": {"context_window": 200000, "reasoning": False},
    "claude-haiku-4-5-20251001": {"context_window": 200000, "reasoning": False},
    # --- GOOGLE ---
    "gemini-3.5-flash": {"context_window": 1048576, "reasoning": True},
    "gemini-3.1-pro": {"context_window": 1048576, "reasoning": True},
    "gemini-3.1-pro-preview": {"context_window": 1048576, "reasoning": True},
    "gemini-3.1-flash-lite": {"context_window": 1048576, "reasoning": True},
    "gemini-3.1-flash-lite-preview": {"context_window": 1048576, "reasoning": True},
    "gemini-3-flash": {"context_window": 1048576, "reasoning": True},
    "gemini-3-flash-preview": {"context_window": 1048576, "reasoning": True},
    "gemini-2.5-pro": {"context_window": 1048576, "reasoning": True},
    "gemini-2.5-flash": {"context_window": 1048576, "reasoning": True},
    "gemini-2.5-flash-lite": {"context_window": 1048576, "reasoning": True},
    "gemini-2.0-pro": {"context_window": 2097152, "reasoning": False},
    "gemini-2.0-flash": {"context_window": 1048576, "reasoning": False},  # shut down 2026-06-01
    "gemini-2.0-flash-lite": {"context_window": 1048576, "reasoning": False},  # shut down 2026-06-01
    "gemini-2.0-flash-thinking": {"context_window": 1048576, "reasoning": True},
    "gemini-1.5-pro": {"context_window": 2097152, "reasoning": False},
    "gemini-1.5-flash": {"context_window": 1048576, "reasoning": False},
    # --- GOOGLE (open models) ---
    "gemma-4-31b": {"context_window": 262144, "reasoning": True},
    "gemma-4-26b-a4b": {"context_window": 262144, "reasoning": True},
    "gemma-4-e4b": {"context_window": 131072, "reasoning": True},
    "gemma-4-e2b": {"context_window": 131072, "reasoning": True},
    "gemma-3-27b-it": {"context_window": 131072, "reasoning": False},
    "gemma-3-12b-it": {"context_window": 131072, "reasoning": False},
    "gemma-3-4b-it": {"context_window": 131072, "reasoning": False},
    "gemma-3-1b-it": {"context_window": 32768, "reasoning": False},
    # --- DEEPSEEK ---
    "deepseek-v4-pro": {"context_window": 1000000, "reasoning": True},
    "deepseek-v4-flash": {"context_window": 1000000, "reasoning": True},
    "deepseek-chat": {"context_window": 128000, "reasoning": False},  # deprecated 2026-07-24
    "deepseek-reasoner": {"context_window": 128000, "reasoning": True},  # deprecated 2026-07-24
    # --- XAI ---
    "grok-4.5": {"context_window": 500000, "reasoning": True},  # configurable reasoning effort
    "grok-4.3": {"context_window": 1000000, "reasoning": False},
    "grok-4.20-0309-reasoning": {"context_window": 1000000, "reasoning": True},
    "grok-4.20-0309-non-reasoning": {"context_window": 1000000, "reasoning": False},
    "grok-4.20-multi-agent-0309": {"context_window": 1000000, "reasoning": False},
    "grok-build-0.1": {"context_window": 256000, "reasoning": False},
    # --- META (open weights) ---
    # Scout advertises 10M; most providers serve far less (often 128k-1M).
    "llama-4-scout": {"context_window": 10485760, "reasoning": False},
    "llama-4-maverick": {"context_window": 1048576, "reasoning": False},
    "llama-3.3-70b-instruct": {"context_window": 131072, "reasoning": False},
    "llama-3.3-8b-instruct": {"context_window": 131072, "reasoning": False},
    "llama-3.2-90b-vision-instruct": {"context_window": 131072, "reasoning": False},
    "llama-3.2-11b-vision-instruct": {"context_window": 131072, "reasoning": False},
    "llama-3.2-3b-instruct": {"context_window": 131072, "reasoning": False},
    "llama-3.2-1b-instruct": {"context_window": 131072, "reasoning": False},
    "llama-3.1-405b-instruct": {"context_window": 131072, "reasoning": False},
    "llama-3.1-70b-instruct": {"context_window": 131072, "reasoning": False},
    "llama-3.1-8b-instruct": {"context_window": 131072, "reasoning": False},
    "llama-3-70b-instruct": {"context_window": 8192, "reasoning": False},
    "llama-3-8b-instruct": {"context_window": 8192, "reasoning": False},
    # Groq-hosted aliases
    "llama-3.3-70b-versatile": {"context_window": 131072, "reasoning": False},
    "llama-3.1-70b-versatile": {"context_window": 131072, "reasoning": False},  # decommissioned on Groq
    "llama-3.1-8b-instant": {"context_window": 131072, "reasoning": False},
    # --- MISTRAL ---
    "open-mistral-7b": {"context_window": 32768, "reasoning": False},
    "open-mixtral-8x7b": {"context_window": 32768, "reasoning": False},
    "open-mistral-nemo": {"context_window": 128000, "reasoning": False},
    "codestral-2508": {"context_window": 256000, "reasoning": False},
    "codestral-2501": {"context_window": 256000, "reasoning": False},
    "mistral-medium-2604": {"context_window": 256000, "reasoning": False},
    "mistral-large-2512": {"context_window": 256000, "reasoning": False},
    "mistral-small-2603": {"context_window": 256000, "reasoning": False},
    "mistral-small-2506": {"context_window": 128000, "reasoning": False},
    "ministral-3-14b-2512": {"context_window": 262144, "reasoning": False},
    "ministral-3-8b-2512": {"context_window": 262144, "reasoning": False},
    "ministral-3-3b-2512": {"context_window": 131072, "reasoning": False},
    "ministral-14b-2512": {"context_window": 256000, "reasoning": False},
    "ministral-8b-2512": {"context_window": 256000, "reasoning": False},
    "ministral-3b-2512": {"context_window": 131072, "reasoning": False},
    "mixtral-8x7b-32768": {"context_window": 32768, "reasoning": False},
    # --- MISTRAL (open weights) ---
    "devstral-2-2512": {"context_window": 256000, "reasoning": False},
    "devstral-small-1.1": {"context_window": 131072, "reasoning": False},
    "devstral-small-2505": {"context_window": 131072, "reasoning": False},
    "magistral-small-2506": {"context_window": 40000, "reasoning": True},
    "mistral-small-3.2-24b": {"context_window": 128000, "reasoning": False},
    "mistral-small-3.1-24b": {"context_window": 128000, "reasoning": False},
    "mistral-small-3": {"context_window": 32768, "reasoning": False},
    "pixtral-large-2411": {"context_window": 128000, "reasoning": False},
    "codestral-mamba": {"context_window": 256000, "reasoning": False},
    "open-mixtral-8x22b": {"context_window": 65536, "reasoning": False},
    # --- MOONSHOT (KIMI) ---
    "kimi-k3": {"context_window": 1000000, "reasoning": True},
    "kimi-k2.7-code": {"context_window": 256000, "reasoning": False},
    "kimi-k2.7-code-highspeed": {"context_window": 256000, "reasoning": False},
    "kimi-k2.6": {"context_window": 256000, "reasoning": True},
    "kimi-k2.5": {"context_window": 256000, "reasoning": True},  # sunset 2026-08-31
    "moonshot-v1-128k": {"context_window": 128000, "reasoning": False},  # sunset 2026-08-31
    "moonshot-v1-32k": {"context_window": 32000, "reasoning": False},  # sunset 2026-08-31
    "moonshot-v1-8k": {"context_window": 8000, "reasoning": False},  # sunset 2026-08-31
    # --- ALIBABA (QWEN, closed weights) ---
    "qwen3.7-max": {"context_window": 1000000, "reasoning": True},
    "qwen3.7-plus": {"context_window": 1000000, "reasoning": True},
    "qwen3.6-flash": {"context_window": 1000000, "reasoning": True},
    # --- ALIBABA (QWEN, open weights, Apache-2.0) ---
    # Qwen3.5/3.6 are 262144 native, extensible to ~1M via YaRN.
    "qwen3.6-27b": {"context_window": 262144, "reasoning": True},
    "qwen3.6-35b-a3b": {"context_window": 262144, "reasoning": True},
    "qwen3.5-397b-a17b": {"context_window": 256000, "reasoning": True},
    "qwen3.5-122b-a10b": {"context_window": 262144, "reasoning": True},
    "qwen3.5-35b-a3b": {"context_window": 262144, "reasoning": True},
    "qwen3.5-27b": {"context_window": 262144, "reasoning": True},
    "qwen3.5-9b": {"context_window": 262144, "reasoning": True},
    "qwen3-235b-a22b": {"context_window": 131072, "reasoning": True},
    "qwen3-30b-a3b": {"context_window": 131072, "reasoning": True},
    "qwen3-32b": {"context_window": 131072, "reasoning": True},
    "qwen3-14b": {"context_window": 131072, "reasoning": True},
    "qwen3-8b": {"context_window": 131072, "reasoning": True},
    "qwen3-coder-480b-a35b": {"context_window": 1048576, "reasoning": False},
    "qwen3-coder-30b-a3b-instruct": {"context_window": 163840, "reasoning": False},
    "qwen2.5-coder-7b-instruct": {"context_window": 131072, "reasoning": False},
    "qwen3-vl-235b-a22b-instruct": {"context_window": 262144, "reasoning": False},
    "qwen3-vl-235b-a22b-thinking": {"context_window": 131072, "reasoning": True},
    "qwen3-vl-32b-instruct": {"context_window": 262144, "reasoning": False},
    "qwen3-vl-30b-a3b-instruct": {"context_window": 262144, "reasoning": False},
    "qwen3-vl-30b-a3b-thinking": {"context_window": 131072, "reasoning": True},
    "qwen3-vl-8b-instruct": {"context_window": 262144, "reasoning": False},
    "qwen3-vl-8b-thinking": {"context_window": 262144, "reasoning": True},
    # --- Z.AI (GLM) ---
    "glm-5.2": {"context_window": 1000000, "reasoning": True},
    "glm-5.1": {"context_window": 200000, "reasoning": True},
}
