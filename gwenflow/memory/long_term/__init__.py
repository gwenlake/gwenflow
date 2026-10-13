"""Long-term markdown memory: what an agent keeps between conversations.

- ``MemoryToolset``: loads the memory files, renders them in the prompt, exposes the tools.
- ``recall``: picks the topic files a request is about, without an LLM call.
- ``MemoryConsolidator``: turns conversation messages into notes and topic files.
- ``MemoryCurator``: an agent that tidies a memory directory.
- ``prompts``: every prompt text, ``templates``: the default file contents.

Storage backends live in :mod:`gwenflow.stores.files`.
"""

from gwenflow.memory.long_term.consolidator import ConsolidationResult, MemoryConsolidator, merge_entities
from gwenflow.memory.long_term.curator import MemoryCurator
from gwenflow.memory.long_term.recall import select_relevant
from gwenflow.memory.long_term.toolset import TOOL_NAMES, FlushReport, MemoryToolset

__all__ = [
    "TOOL_NAMES",
    "ConsolidationResult",
    "FlushReport",
    "MemoryConsolidator",
    "MemoryCurator",
    "MemoryToolset",
    "merge_entities",
    "select_relevant",
]
