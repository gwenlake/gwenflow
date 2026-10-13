"""Versioned stores for markdown memory files (``/memories/...`` virtual paths).

``MemoryStore`` is the protocol; ``InMemoryMemoryStore``, ``LocalFileMemoryStore``
and ``S3MemoryStore`` are the reference backends. Applications plug their own
(a database table, for instance) by implementing the protocol.
"""

from gwenflow.stores.files.base import (
    MEMORY_ROOT,
    MemoryConflictError,
    MemoryFile,
    MemoryPathError,
    MemoryStore,
    Version,
    description_from_content,
    normalize_memory_path,
    normalize_memory_prefix,
)
from gwenflow.stores.files.in_memory import InMemoryMemoryStore
from gwenflow.stores.files.local import LocalFileMemoryStore
from gwenflow.stores.files.s3 import S3MemoryStore

__all__ = [
    "MEMORY_ROOT",
    "InMemoryMemoryStore",
    "LocalFileMemoryStore",
    "MemoryConflictError",
    "MemoryFile",
    "MemoryPathError",
    "MemoryStore",
    "S3MemoryStore",
    "Version",
    "description_from_content",
    "normalize_memory_path",
    "normalize_memory_prefix",
]
