"""Built-in LLM providers and tools, addressable by name.

Shared by the CLI and by flows, whose YAML refers to models as
`provider/model-name` and to tools by their short name.
"""

import importlib
from typing import Any

PROVIDERS: dict[str, tuple[str, str]] = {
    "openai": ("gwenflow.llms.openai", "ChatOpenAI"),
    "anthropic": ("gwenflow.llms.anthropic", "ChatAnthropic"),
    "azure": ("gwenflow.llms.azure", "ChatAzureOpenAI"),
    "google": ("gwenflow.llms.google", "ChatGoogle"),
    "gemini": ("gwenflow.llms.google", "ChatGoogle"),
    "mistral": ("gwenflow.llms.mistral", "ChatMistral"),
    "ollama": ("gwenflow.llms.ollama", "ChatOllama"),
    "deepseek": ("gwenflow.llms.deepseek", "ChatDeepSeek"),
    "gwenlake": ("gwenflow.llms.gwenlake", "ChatGwenlake"),
}

TOOLS: dict[str, tuple[str, str]] = {
    "wikipedia": ("gwenflow.tools.wikipedia", "WikipediaTool"),
    "duckduckgo": ("gwenflow.tools.duckduckgo", "DuckDuckGoSearchTool"),
    "duckduckgo-news": ("gwenflow.tools.duckduckgo", "DuckDuckGoNewsTool"),
    "tavily": ("gwenflow.tools.tavily", "TavilyWebSearchTool"),
    "python": ("gwenflow.tools.python", "PythonCodeTool"),
    "shell": ("gwenflow.tools.shell", "ShellTool"),
    "website": ("gwenflow.tools.website", "WebsiteReaderTool"),
    "pdf": ("gwenflow.tools.pdf", "PDFReaderTool"),
    "yahoo-stock": ("gwenflow.tools.yahoofinance", "YahooFinanceStock"),
    "yahoo-news": ("gwenflow.tools.yahoofinance", "YahooFinanceNews"),
    "yahoo-screen": ("gwenflow.tools.yahoofinance", "YahooFinanceScreen"),
}

DEFAULT_MODEL = "openai/gpt-4o-mini"


def _load(entry: tuple[str, str]) -> Any:
    module_path, cls_name = entry
    return getattr(importlib.import_module(module_path), cls_name)


def make_llm(model: str = DEFAULT_MODEL, **kwargs: Any) -> Any:
    """`provider/model-name` (or a bare model name, OpenAI) -> LLM instance."""
    provider, _, model_name = model.partition("/")
    if not model_name:
        provider, model_name = "openai", provider
    entry = PROVIDERS.get(provider.lower())
    if entry is None:
        raise ValueError(f"Unknown provider '{provider}'. Choose from: {', '.join(PROVIDERS)}")
    return _load(entry)(model=model_name, **kwargs)


def make_tools(names: Any) -> list:
    tools = []
    for name in names or []:
        entry = TOOLS.get(str(name).lower())
        if entry is None:
            raise ValueError(f"Unknown tool '{name}'. Available: {', '.join(TOOLS)}")
        tools.append(_load(entry)())
    return tools


def model_id(llm: Any) -> str:
    """Inverse of `make_llm`: the `provider/model-name` of a built-in LLM instance."""
    cls_name = type(llm).__name__
    for provider, (_, name) in PROVIDERS.items():
        if name == cls_name:
            return f"{provider}/{llm.model}"
    raise ValueError(f"{cls_name} is not a built-in provider: it cannot be referenced from a flow.")


def tool_key(tool: Any) -> str:
    """Inverse of `make_tools` for one built-in tool instance."""
    cls_name = type(tool).__name__
    for key, (_, name) in TOOLS.items():
        if name == cls_name:
            return key
    raise ValueError(f"{cls_name} is not a built-in tool: it cannot be referenced from a flow.")
