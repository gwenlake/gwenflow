from typing import TypedDict


class ModelInfo(TypedDict):
    context_window: int
    reasoning: bool


MODELS: dict[str, ModelInfo] = {

    # --- OPENAI ---
    "gpt-5.4-pro":              {"context_window": 1050000, "reasoning": True},
    "gpt-5.4":                  {"context_window": 1050000, "reasoning": True},
    "gpt-5.4-mini":             {"context_window":  400000, "reasoning": True},
    "gpt-5.4-nano":             {"context_window":  400000, "reasoning": True},
    "gpt-5.2-pro":              {"context_window":  400000, "reasoning": True},
    "gpt-5.2":                  {"context_window":  400000, "reasoning": True},
    "gpt-5":                    {"context_window":  400000, "reasoning": True},
    "gpt-5-mini":               {"context_window":  400000, "reasoning": True},
    "gpt-5-nano":               {"context_window":  400000, "reasoning": True},
    "gpt-4.5":                  {"context_window":  128000, "reasoning": False},
    "gpt-4.1":                  {"context_window": 1047576, "reasoning": False},
    "gpt-4.1-mini":             {"context_window": 1047576, "reasoning": False},
    "gpt-4.1-nano":             {"context_window": 1047576, "reasoning": False},
    "gpt-4o":                   {"context_window":  128000, "reasoning": False},
    "gpt-4o-mini":              {"context_window":  128000, "reasoning": False},
    "o1":                       {"context_window":  200000, "reasoning": True},
    "o1-preview":               {"context_window":  128000, "reasoning": True},
    "o1-mini":                  {"context_window":  128000, "reasoning": True},
    "o3":                       {"context_window":  200000, "reasoning": True},
    "o3-pro":                   {"context_window":  200000, "reasoning": True},
    "o3-mini":                  {"context_window":  200000, "reasoning": True},
    "o3-mini-high":             {"context_window":  128000, "reasoning": True},
    "o4-mini":                  {"context_window":  128000, "reasoning": True},

    # --- ANTHROPIC ---
    "claude-opus-4-7":          {"context_window": 1000000, "reasoning": True},
    "claude-opus-4-6":          {"context_window": 1000000, "reasoning": True},
    "claude-opus-4-5":          {"context_window":  200000, "reasoning": True},
    "claude-opus-4-1":          {"context_window":  200000, "reasoning": True},
    "claude-sonnet-4-6":        {"context_window": 1000000, "reasoning": False},
    "claude-sonnet-4-5":        {"context_window":  200000, "reasoning": False},
    "claude-haiku-4-5":         {"context_window":  200000, "reasoning": False},
    "claude-haiku-4-5-20251001":{"context_window":  200000, "reasoning": False},

    # --- GOOGLE ---
    "gemini-3.1-pro-preview":         {"context_window": 1048576, "reasoning": True},
    "gemini-3.1-flash-lite-preview":  {"context_window": 1048576, "reasoning": True},
    "gemini-3-flash-preview":         {"context_window": 1048576, "reasoning": True},
    "gemini-2.5-pro":                 {"context_window": 1048576, "reasoning": True},
    "gemini-2.5-flash":               {"context_window": 1048576, "reasoning": True},
    "gemini-2.5-flash-lite":          {"context_window": 1048576, "reasoning": True},
    "gemini-2.0-pro":                 {"context_window": 2097152, "reasoning": False},
    "gemini-2.0-flash":               {"context_window": 1048576, "reasoning": False},
    "gemini-2.0-flash-lite":          {"context_window": 1048576, "reasoning": False},
    "gemini-2.0-flash-thinking":      {"context_window": 1048576, "reasoning": True},
    "gemini-1.5-pro":                 {"context_window": 2097152, "reasoning": False},
    "gemini-1.5-flash":               {"context_window": 1048576, "reasoning": False},

    # --- DEEPSEEK ---
    "deepseek-chat":            {"context_window":  128000, "reasoning": False},
    "deepseek-r1":              {"context_window":  128000, "reasoning": True},

    # --- META ---
    "llama-3.1-70b-versatile":  {"context_window":  131072, "reasoning": False},
    "llama-3.1-8b-instant":     {"context_window":  131072, "reasoning": False},

    # --- MISTRAL ---
    "open-mistral-7b":          {"context_window":   32768, "reasoning": False},
    "open-mixtral-8x7b":        {"context_window":   32768, "reasoning": False},
    "open-mistral-nemo":        {"context_window":  128000, "reasoning": False},
    "codestral-2501":           {"context_window":  256000, "reasoning": False},
    "mistral-large-2512":       {"context_window":  256000, "reasoning": False},
    "mistral-small-2603":       {"context_window":  256000, "reasoning": False},
    "mistral-small-2506":       {"context_window":  128000, "reasoning": False},
    "ministral-14b-2512":       {"context_window":  256000, "reasoning": False},
    "ministral-8b-2512":        {"context_window":  256000, "reasoning": False},
    "ministral-3b-2512":        {"context_window":  256000, "reasoning": False},
    "mixtral-8x7b-32768":       {"context_window":   32768, "reasoning": False},

}
