"""Chat backends.

`ChatAnthropic`, `ChatGoogle` and `ChatMistral` are resolved lazily (PEP 562):
each one pulls a vendor SDK worth tens of MiB of RSS, which is dead weight for
a deployment that only talks to one provider. They stay importable from this
package as before -- the SDK is only loaded on first attribute access.

The other backends subclass `ChatOpenAI` and need no extra dependency, so they
are imported eagerly.
"""

import importlib
from typing import TYPE_CHECKING, Any

from gwenflow.llms.azure import ChatAzureOpenAI
from gwenflow.llms.base import ChatBase
from gwenflow.llms.deepseek import ChatDeepSeek
from gwenflow.llms.gwenlake import ChatGwenlake
from gwenflow.llms.ollama import ChatOllama
from gwenflow.llms.openai import ChatOpenAI

if TYPE_CHECKING:
    from gwenflow.llms.anthropic import ChatAnthropic
    from gwenflow.llms.google import ChatGoogle
    from gwenflow.llms.mistral import ChatMistral

_LAZY_BACKENDS = {
    "ChatAnthropic": "gwenflow.llms.anthropic",
    "ChatGoogle": "gwenflow.llms.google",
    "ChatMistral": "gwenflow.llms.mistral",
}


def __getattr__(name: str) -> Any:
    module_path = _LAZY_BACKENDS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_path), name)
    globals()[name] = value  # cache it so __getattr__ runs only once
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_BACKENDS))


__all__ = [
    "ChatBase",
    "ChatAnthropic",
    "ChatOpenAI",
    "ChatAzureOpenAI",
    "ChatGoogle",
    "ChatMistral",
    "ChatGwenlake",
    "ChatOllama",
    "ChatDeepSeek",
]
