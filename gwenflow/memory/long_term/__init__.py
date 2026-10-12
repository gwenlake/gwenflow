"""Long-term markdown memory: what an agent keeps between conversations.

- ``MemoryToolset``: loads the memory files, renders them in the prompt, exposes the tools.
- ``recall``: picks the topic files a request is about, without an LLM call.
- ``MemoryConsolidator``: turns conversation messages into notes and topic files.
- ``build_memory_curator``: an agent that tidies a memory directory.

Storage backends live in :mod:`gwenflow.stores.files`.
"""

from gwenflow.memory.long_term.consolidator import ConsolidationResult, MemoryConsolidator, merge_entities
from gwenflow.memory.long_term.curator import build_memory_curator
from gwenflow.memory.long_term.recall import select_relevant
from gwenflow.memory.long_term.toolset import FlushReport, MemoryToolset

__all__ = [
    "ConsolidationResult",
    "FlushReport",
    "MemoryConsolidator",
    "MemoryToolset",
    "build_memory_curator",
    "merge_entities",
    "select_relevant",
]
