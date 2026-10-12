"""Storage backends for long-term markdown memory.

Memory files live under a virtual ``/memories`` root. A store maps those
canonical paths onto whatever it persists to (a dict, a directory, an S3
bucket, a database table). Every path that reaches a store goes through
:func:`normalize_memory_path` first, so a store never sees ``..`` segments or
paths outside the root.

Writes carry an optional ``expected_version`` for optimistic locking: several
processes may read the same file, work on it, and write it back; only the
first writer wins, the others get :class:`MemoryConflictError` and can reload
and retry. The version is opaque to callers (an integer, an mtime, an ETag).

Each file also carries ``metadata``: a small JSON document the model never
sees, for application state (entities, job watermarks, ...). Reference
backends keep it in a sidecar next to the file.
"""

from __future__ import annotations

import asyncio
import json
import os
import posixpath
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from gwenflow.exceptions import GwenflowException, UserError
from gwenflow.utils.markdown import parse_frontmatter

MEMORY_ROOT = "/memories"

Version = int | str | None

_SEGMENT_RE = re.compile(r"^[A-Za-z0-9._-]+$")


class MemoryPathError(UserError):
    """The path is outside ``/memories``, not a ``.md`` file, or malformed."""


class MemoryConflictError(GwenflowException):
    """The stored version differs from ``expected_version``: someone else wrote first."""

    def __init__(self, path: str, expected: Version, actual: Version):
        self.path = path
        self.expected = expected
        self.actual = actual
        super().__init__(f"Memory file {path} changed concurrently (expected version {expected}, found {actual})")


def normalize_memory_path(path: str) -> str:
    """Return the canonical form of a memory path or raise :class:`MemoryPathError`.

    Rules: must start with ``/memories/``, every segment is made of
    ``[A-Za-z0-9._-]`` and is neither ``.`` nor ``..``, and the file name ends
    with ``.md``. ``posixpath.normpath`` runs first so ``/memories/a/../b.md``
    collapses to ``/memories/b.md`` and is then accepted, while
    ``/memories/../etc/x.md`` collapses outside the root and is rejected.
    """
    if not isinstance(path, str) or not path.strip():
        raise MemoryPathError("Memory path is empty")
    candidate = path.strip()
    if "%" in candidate or "\\" in candidate:
        raise MemoryPathError(f"Memory path contains illegal characters: {path!r}")
    normalized = posixpath.normpath(candidate)
    if not normalized.startswith(MEMORY_ROOT + "/"):
        raise MemoryPathError(f"Memory path must be under {MEMORY_ROOT}/: {path!r}")
    relative = normalized[len(MEMORY_ROOT) + 1 :]
    segments = relative.split("/")
    for segment in segments:
        if segment in ("", ".", "..") or not _SEGMENT_RE.match(segment):
            raise MemoryPathError(f"Memory path has an illegal segment {segment!r}: {path!r}")
    if not segments[-1].endswith(".md") or segments[-1] == ".md":
        raise MemoryPathError(f"Memory files must be .md files: {path!r}")
    return normalized


def normalize_memory_prefix(prefix: str) -> str:
    """Canonical directory prefix (``/memories/user/``) used by ``alist``."""
    if not isinstance(prefix, str) or not prefix.strip():
        raise MemoryPathError("Memory prefix is empty")
    candidate = posixpath.normpath(prefix.strip())
    if candidate != MEMORY_ROOT and not candidate.startswith(MEMORY_ROOT + "/"):
        raise MemoryPathError(f"Memory prefix must be under {MEMORY_ROOT}: {prefix!r}")
    for segment in candidate[len(MEMORY_ROOT) :].split("/"):
        if segment and (segment in (".", "..") or not _SEGMENT_RE.match(segment)):
            raise MemoryPathError(f"Memory prefix has an illegal segment {segment!r}: {prefix!r}")
    return candidate.rstrip("/") + "/"


@dataclass
class MemoryFile:
    """One memory file as a store returns it.

    ``version`` is ``None`` for a file that does not exist yet (a template the
    toolset rendered, for instance). ``description`` is the frontmatter
    description when the store keeps it apart; otherwise it is derived from
    ``content``. ``content`` is ``None`` in index listings.
    """

    path: str
    content: str | None = None
    version: Version = None
    description: str | None = None
    last_read: datetime | None = None
    updated_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def exists(self) -> bool:
        return self.version is not None

    @property
    def frontmatter(self) -> dict[str, Any]:
        if not self.content:
            return {}
        return parse_frontmatter(self.content)[0]

    @property
    def body(self) -> str:
        if not self.content:
            return ""
        return parse_frontmatter(self.content)[1]

    def resolved_description(self) -> str:
        if self.description:
            return self.description
        description = self.frontmatter.get("description")
        return str(description).strip() if description else ""


def description_from_content(content: str | None) -> str | None:
    if not content:
        return None
    description = parse_frontmatter(content)[0].get("description")
    return str(description).strip() if description else None


@runtime_checkable
class MemoryStore(Protocol):
    """Storage backend for markdown memory files.

    Async for everything the toolset calls around a run (``aload`` /
    ``aflush``), plus one sync ``read`` used by the ``read_memory`` tool: tools
    run in a worker thread (``Agent.run_tool`` is sync), so a backend with an
    async driver implements ``read`` by handing the coroutine back to the main
    event loop (``asyncio.run_coroutine_threadsafe``).
    """

    async def aread(self, path: str) -> MemoryFile | None: ...

    async def aread_many(self, paths: list[str]) -> list[MemoryFile]:
        """Existing files among ``paths`` (missing ones are simply absent)."""
        ...

    async def alist(self, prefix: str) -> list[MemoryFile]:
        """Index entries (no ``content``) for every file whose path starts with ``prefix``."""
        ...

    async def awrite(self, path: str, content: str, expected_version: Version = None) -> MemoryFile:
        """Create or replace ``path``.

        ``expected_version=None`` means "the file must not exist yet";
        otherwise the stored version must match. Raise
        :class:`MemoryConflictError` in both failure cases.
        """
        ...

    async def adelete(self, path: str) -> None: ...

    async def atouch_read(self, paths: list[str]) -> None:
        """Record that ``paths`` were read (for forgetting policies). May be a no-op."""
        ...

    async def aupdate_metadata(self, path: str, patch: dict[str, Any]) -> None:
        """Merge ``patch`` into the file's metadata without changing its content or version."""
        ...

    def read(self, path: str) -> MemoryFile | None:
        """Sync read used from a tool thread."""
        ...


# ---------------------------------------------------------------------------
# Sidecar helpers shared by the file-like backends
# ---------------------------------------------------------------------------

SIDECAR_SUFFIX = ".meta.json"


def _dump_sidecar(last_read: datetime | None, metadata: dict[str, Any]) -> str:
    return json.dumps(
        {"last_read": last_read.isoformat() if last_read else None, "metadata": metadata},
        ensure_ascii=False,
        default=str,
    )


def _parse_sidecar(raw: str | None) -> tuple[datetime | None, dict[str, Any]]:
    if not raw:
        return None, {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None, {}
    last_read = data.get("last_read")
    metadata = data.get("metadata") or {}
    try:
        parsed = datetime.fromisoformat(last_read) if last_read else None
    except ValueError:
        parsed = None
    return parsed, metadata if isinstance(metadata, dict) else {}


# ---------------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------------


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


class LocalFileMemoryStore:
    """Real ``.md`` files under ``base_dir`` (``/memories/user/MEMORY.md`` -> ``base_dir/user/MEMORY.md``).

    The version is the file's ``st_mtime_ns``, which is enough to detect a
    concurrent rewrite on one machine. ``last_read`` and ``metadata`` live in a
    ``<file>.meta.json`` sidecar. Every resolved path is checked to stay under
    ``base_dir`` (symlinks included), like the local file-system tools.
    """

    def __init__(self, base_dir: str | os.PathLike[str]) -> None:
        self.base_dir = Path(base_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _to_disk(self, path: str) -> Path:
        normalized = normalize_memory_path(path)
        disk = (self.base_dir / normalized[len(MEMORY_ROOT) + 1 :]).resolve()
        try:
            disk.relative_to(self.base_dir)
        except ValueError as e:
            raise MemoryPathError(f"Memory path escapes the memory directory: {path!r}") from e
        return disk

    def _to_virtual(self, disk: Path) -> str:
        return MEMORY_ROOT + "/" + disk.relative_to(self.base_dir).as_posix()

    @staticmethod
    def _sidecar(disk: Path) -> Path:
        return disk.with_name(disk.name + SIDECAR_SUFFIX)

    @staticmethod
    def _version(disk: Path) -> int | None:
        try:
            return disk.stat().st_mtime_ns
        except FileNotFoundError:
            return None

    def _load(self, disk: Path, *, with_content: bool) -> MemoryFile | None:
        if not disk.is_file():
            return None
        content = disk.read_text(encoding="utf-8")
        sidecar = self._sidecar(disk)
        last_read, metadata = _parse_sidecar(sidecar.read_text(encoding="utf-8") if sidecar.is_file() else None)
        stat = disk.stat()
        return MemoryFile(
            path=self._to_virtual(disk),
            content=content if with_content else None,
            version=stat.st_mtime_ns,
            description=description_from_content(content),
            last_read=last_read,
            updated_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
            metadata=metadata,
        )

    def _save_sidecar(self, disk: Path, last_read: datetime | None, metadata: dict[str, Any]) -> None:
        self._sidecar(disk).write_text(_dump_sidecar(last_read, metadata), encoding="utf-8")

    def read(self, path: str) -> MemoryFile | None:
        disk = self._to_disk(path)
        with self._lock:
            return self._load(disk, with_content=True)

    def write(self, path: str, content: str, expected_version: Version = None) -> MemoryFile:
        disk = self._to_disk(path)
        normalized = normalize_memory_path(path)
        with self._lock:
            actual = self._version(disk)
            if actual != expected_version:
                raise MemoryConflictError(normalized, expected_version, actual)
            disk.parent.mkdir(parents=True, exist_ok=True)
            disk.write_text(content, encoding="utf-8")
            new_version = self._version(disk)
            if new_version == actual:
                # Same mtime tick: bump it so the version actually changes.
                os.utime(disk, ns=(disk.stat().st_atime_ns, actual + 1))
            return self._load(disk, with_content=True)

    def delete(self, path: str) -> None:
        disk = self._to_disk(path)
        with self._lock:
            if disk.is_file():
                disk.unlink()
            sidecar = self._sidecar(disk)
            if sidecar.is_file():
                sidecar.unlink()

    def list(self, prefix: str) -> list[MemoryFile]:
        prefix = normalize_memory_prefix(prefix)
        root = (self.base_dir / prefix[len(MEMORY_ROOT) + 1 :]).resolve()
        try:
            root.relative_to(self.base_dir)
        except ValueError as e:
            raise MemoryPathError(f"Memory prefix escapes the memory directory: {prefix!r}") from e
        if not root.is_dir():
            return []
        files: list[MemoryFile] = []
        with self._lock:
            for disk in sorted(root.rglob("*.md")):
                if not disk.is_file():
                    continue
                try:
                    normalize_memory_path(self._to_virtual(disk))
                except MemoryPathError:
                    continue
                loaded = self._load(disk, with_content=False)
                if loaded is not None:
                    files.append(loaded)
        return files

    def update_metadata(self, path: str, patch: dict[str, Any]) -> None:
        disk = self._to_disk(path)
        with self._lock:
            current = self._load(disk, with_content=False)
            if current is None:
                return
            self._save_sidecar(disk, current.last_read, {**current.metadata, **patch})

    async def aread(self, path: str) -> MemoryFile | None:
        return await asyncio.to_thread(self.read, path)

    async def aread_many(self, paths: list[str]) -> list[MemoryFile]:
        files = [await self.aread(p) for p in paths]
        return [f for f in files if f is not None]

    async def alist(self, prefix: str) -> list[MemoryFile]:
        return await asyncio.to_thread(self.list, prefix)

    async def awrite(self, path: str, content: str, expected_version: Version = None) -> MemoryFile:
        return await asyncio.to_thread(self.write, path, content, expected_version)

    async def adelete(self, path: str) -> None:
        await asyncio.to_thread(self.delete, path)

    async def atouch_read(self, paths: list[str]) -> None:
        now = datetime.now(timezone.utc)

        def touch() -> None:
            with self._lock:
                for p in paths:
                    disk = self._to_disk(p)
                    current = self._load(disk, with_content=False)
                    if current is not None:
                        self._save_sidecar(disk, now, current.metadata)

        await asyncio.to_thread(touch)

    async def aupdate_metadata(self, path: str, patch: dict[str, Any]) -> None:
        await asyncio.to_thread(self.update_metadata, path, patch)
