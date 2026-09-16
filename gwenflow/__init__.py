"""Gwenflow public API.

The vendor-SDK-backed chat backends are resolved lazily (PEP 562) so that
`import gwenflow` does not drag in every provider's SDK. See
`gwenflow.llms.__init__` for the rationale.
"""

import importlib
from typing import TYPE_CHECKING, Any

from gwenflow.agents import Agent
from gwenflow.exceptions import (
    GwenflowException,
    MaxTurnsExceeded,
    ModelBehaviorError,
    UserError,
)
from gwenflow.flows import AutoFlow, Flow, FlowRunner
from gwenflow.llms import (
    ChatAzureOpenAI,
    ChatDeepSeek,
    ChatGwenlake,
    ChatOllama,
    ChatOpenAI,
)
from gwenflow.logger import logger, set_log_level_to_debug
from gwenflow.retriever import Retriever
from gwenflow.telemetry import Telemetry
from gwenflow.tools import BaseTool, Tool
from gwenflow.types import Document, Message

if TYPE_CHECKING:
    from gwenflow.llms import ChatAnthropic, ChatGoogle, ChatMistral
    from gwenflow.readers import SimpleDirectoryReader

_LAZY_EXPORTS = {
    "ChatAnthropic": "gwenflow.llms",
    "ChatGoogle": "gwenflow.llms",
    "ChatMistral": "gwenflow.llms",
    "SimpleDirectoryReader": "gwenflow.readers",
}


def __getattr__(name: str) -> Any:
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_path), name)
    globals()[name] = value  # cache it so __getattr__ runs only once
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


__all__ = [
    "logger",
    "set_log_level_to_debug",
    "GwenflowException",
    "MaxTurnsExceeded",
    "ModelBehaviorError",
    "UserError",
    "ChatGwenlake",
    "ChatOpenAI",
    "ChatAzureOpenAI",
    "ChatAnthropic",
    "ChatGoogle",
    "ChatMistral",
    "ChatDeepSeek",
    "ChatOllama",
    "Document",
    "Message",
    "SimpleDirectoryReader",
    "Retriever",
    "Agent",
    "BaseTool",
    "Tool",
    "Flow",
    "FlowRunner",
    "AutoFlow",
    "Telemetry",
]
