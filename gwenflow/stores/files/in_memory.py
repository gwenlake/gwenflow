"""Dict-backed memory store: tests, dev and short-lived agents."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from gwenflow.stores.files.base import (
    MemoryConflictError,
    MemoryFile,
    Version,
    description_from_content,
    normalize_memory_path,
    normalize_memory_prefix,
)


@dataclass
class _Entry:
    content: str
    version: int
    updated_at: datetime
    last_read: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class InMemoryMemoryStore:
    """Dict-backed store for tests, dev and short-lived agents. Versions start at 1.

    Process-local: with several workers each has its own copy, and a restart
    empties it.
    """

    def __init__(self) -> None:
        self._files: dict[str, _Entry] = {}
        self._lock = threading.Lock()

    def _to_file(self, path: str, entry: _Entry, *, with_content: bool = True) -> MemoryFile:
        return MemoryFile(
            path=path,
            content=entry.content if with_content else None,
            version=entry.version,
            description=description_from_content(entry.content),
            last_read=entry.last_read,
            updated_at=entry.updated_at,
            metadata=dict(entry.metadata),
        )

    # -- sync core -----------------------------------------------------------

    def read(self, path: str) -> MemoryFile | None:
        path = normalize_memory_path(path)
        with self._lock:
            entry = self._files.get(path)
            return self._to_file(path, entry) if entry else None

    def write(self, path: str, content: str, expected_version: Version = None) -> MemoryFile:
        path = normalize_memory_path(path)
        with self._lock:
            current = self._files.get(path)
            actual = current.version if current else None
            if actual != expected_version:
                raise MemoryConflictError(path, expected_version, actual)
            entry = _Entry(
                content=content,
                version=(actual or 0) + 1,
                updated_at=datetime.now(timezone.utc),
                last_read=current.last_read if current else None,
                metadata=dict(current.metadata) if current else {},
            )
            self._files[path] = entry
            return self._to_file(path, entry)

    def delete(self, path: str) -> None:
        path = normalize_memory_path(path)
        with self._lock:
            self._files.pop(path, None)

    def list(self, prefix: str) -> list[MemoryFile]:
        prefix = normalize_memory_prefix(prefix)
        with self._lock:
            return [
                self._to_file(path, entry, with_content=False)
                for path, entry in sorted(self._files.items())
                if path.startswith(prefix)
            ]

    def update_metadata(self, path: str, patch: dict[str, Any]) -> None:
        path = normalize_memory_path(path)
        with self._lock:
            entry = self._files.get(path)
            if entry is not None:
                entry.metadata = {**entry.metadata, **patch}

    # -- async facade --------------------------------------------------------

    async def aread(self, path: str) -> MemoryFile | None:
        return self.read(path)

    async def aread_many(self, paths: list[str]) -> list[MemoryFile]:
        files = [self.read(p) for p in paths]
        return [f for f in files if f is not None]

    async def alist(self, prefix: str) -> list[MemoryFile]:
        return self.list(prefix)

    async def awrite(self, path: str, content: str, expected_version: Version = None) -> MemoryFile:
        return self.write(path, content, expected_version)

    async def adelete(self, path: str) -> None:
        self.delete(path)

    async def atouch_read(self, paths: list[str]) -> None:
        now = datetime.now(timezone.utc)
        with self._lock:
            for p in paths:
                entry = self._files.get(normalize_memory_path(p))
                if entry is not None:
                    entry.last_read = now

    async def aupdate_metadata(self, path: str, patch: dict[str, Any]) -> None:
        self.update_metadata(path, patch)
