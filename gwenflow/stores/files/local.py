"""Memory store on real ``.md`` files in a local directory."""

from __future__ import annotations

import asyncio
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gwenflow.stores.files.base import (
    MEMORY_ROOT,
    SIDECAR_SUFFIX,
    MemoryConflictError,
    MemoryFile,
    MemoryPathError,
    Version,
    _dump_sidecar,
    _parse_sidecar,
    description_from_content,
    normalize_memory_path,
    normalize_memory_prefix,
)


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
