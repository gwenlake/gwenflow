"""Agents.

`CodingAgent` and its helpers are resolved lazily (PEP 562): they import a
concrete toolset (shell, file editing, website reading, ...) whose dependencies
are irrelevant to a process that only builds a plain `Agent`.
"""

import importlib
from typing import TYPE_CHECKING, Any

from gwenflow.agents.agent import Agent

if TYPE_CHECKING:
    from gwenflow.agents.coding_agent import (
        CODING_AGENT_INSTRUCTIONS,
        CodingAgent,
        build_coding_tools,
    )

_LAZY_AGENTS = {
    "CODING_AGENT_INSTRUCTIONS": "gwenflow.agents.coding_agent",
    "CodingAgent": "gwenflow.agents.coding_agent",
    "build_coding_tools": "gwenflow.agents.coding_agent",
}


def __getattr__(name: str) -> Any:
    module_path = _LAZY_AGENTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_path), name)
    globals()[name] = value  # cache it so __getattr__ runs only once
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_AGENTS))


__all__ = [
    "Agent",
    "CodingAgent",
    "build_coding_tools",
    "CODING_AGENT_INSTRUCTIONS",
]
