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

import json
import posixpath
import re
from dataclasses import dataclass, field
from datetime import datetime
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
