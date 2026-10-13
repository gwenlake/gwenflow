"""S3-backed memory store (AWS S3, Garage, MinIO, any S3-compatible endpoint).

``/memories/user/MEMORY.md`` -> ``s3://<bucket>/<prefix>/user/MEMORY.md``.
The version is the object's ETag. Every write compares the current ETag first
(compare-then-put), which closes the race window to a few milliseconds without
making it zero; with ``conditional_writes=True`` the precondition headers are
sent as well (``If-None-Match: *`` on create, ``If-Match: <etag>`` on update),
which is atomic on servers that honour them, AWS S3 for instance. Garage
accepts the headers and ignores them, hence the default off. ``last_read``
and ``metadata`` live in a ``<key>.meta.json`` sidecar.

Needs ``boto3`` (a gwenflow dependency). All calls are blocking and run in a
worker thread behind the async facade.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

from gwenflow.logger import logger
from gwenflow.stores.files.base import (
    MEMORY_ROOT,
    SIDECAR_SUFFIX,
    MemoryConflictError,
    MemoryFile,
    Version,
    _dump_sidecar,
    _parse_sidecar,
    description_from_content,
    normalize_memory_path,
    normalize_memory_prefix,
)

_NOT_FOUND = {"404", "NoSuchKey", "NotFound"}
_PRECONDITION = {"412", "PreconditionFailed"}
_UNSUPPORTED = {"501", "NotImplemented", "InvalidArgument", "InvalidRequest"}


def _error_code(error: Exception) -> str:
    response = getattr(error, "response", None) or {}
    return str(
        (response.get("Error") or {}).get("Code") or response.get("ResponseMetadata", {}).get("HTTPStatusCode", "")
    )


class S3MemoryStore:
    def __init__(
        self,
        bucket: str,
        prefix: str = "memories",
        *,
        client: Any = None,
        endpoint_url: str | None = None,
        region_name: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        conditional_writes: bool = False,
    ) -> None:
        """``client`` is a ready boto3 S3 client; otherwise one is built from the other arguments.

        ``conditional_writes=True`` adds ``If-Match`` / ``If-None-Match`` to every
        write on top of the compare-then-put check; only enable it on servers
        that honour them (AWS S3), others accept and ignore the headers.
        """
        if client is None:
            import boto3

            client = boto3.client(
                "s3",
                endpoint_url=endpoint_url,
                region_name=region_name,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
            )
        self.client = client
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self._conditional_writes = conditional_writes
        self._pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="s3-memory")

    # ------------------------------------------------------------------
    # key mapping
    # ------------------------------------------------------------------

    def _key(self, path: str) -> str:
        normalized = normalize_memory_path(path)
        return (
            f"{self.prefix}/{normalized[len(MEMORY_ROOT) + 1 :]}" if self.prefix else normalized[len(MEMORY_ROOT) + 1 :]
        )

    def _key_prefix(self, prefix: str) -> str:
        normalized = normalize_memory_prefix(prefix)
        relative = normalized[len(MEMORY_ROOT) + 1 :]
        return f"{self.prefix}/{relative}" if self.prefix else relative

    def _path(self, key: str) -> str:
        relative = key[len(self.prefix) + 1 :] if self.prefix else key
        return f"{MEMORY_ROOT}/{relative}"

    # ------------------------------------------------------------------
    # raw S3 helpers (blocking)
    # ------------------------------------------------------------------

    def _get(self, key: str) -> tuple[str, str, datetime] | None:
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=key)
        except Exception as e:  # noqa: BLE001 — translate "missing" only
            if _error_code(e) in _NOT_FOUND:
                return None
            raise
        body = response["Body"].read().decode("utf-8")
        last_modified = response.get("LastModified") or datetime.now(timezone.utc)
        return body, str(response.get("ETag", "")).strip('"'), last_modified

    def _head_etag(self, key: str) -> str | None:
        try:
            response = self.client.head_object(Bucket=self.bucket, Key=key)
        except Exception as e:  # noqa: BLE001
            if _error_code(e) in _NOT_FOUND:
                return None
            raise
        return str(response.get("ETag", "")).strip('"')

    def _put(self, key: str, body: str, *, expected_etag: str | None, must_not_exist: bool) -> str:
        """Write ``body`` and return the new ETag.

        The current ETag is always compared first (compare-then-put): some
        S3-compatible servers accept ``If-Match`` and silently ignore it, so
        acceptance proves nothing. With ``conditional_writes=True`` the
        precondition headers are sent as well, which makes the check atomic on
        servers that honour them (AWS S3).
        """
        actual = self._head_etag(key)
        if (must_not_exist and actual is not None) or (not must_not_exist and actual != expected_etag):
            raise MemoryConflictError(self._path(key), expected_etag, actual)
        params: dict[str, Any] = {
            "Bucket": self.bucket,
            "Key": key,
            "Body": body.encode("utf-8"),
            "ContentType": "text/markdown",
        }
        if self._conditional_writes:
            if must_not_exist:
                params["IfNoneMatch"] = "*"
            elif expected_etag:
                params["IfMatch"] = expected_etag
            try:
                response = self.client.put_object(**params)
                return str(response.get("ETag", "")).strip('"')
            except Exception as e:  # noqa: BLE001
                code = _error_code(e)
                if code in _PRECONDITION:
                    raise MemoryConflictError(self._path(key), expected_etag, self._head_etag(key)) from e
                if code not in _UNSUPPORTED and not isinstance(e, TypeError):
                    raise
                logger.info("[MEMORY] S3 endpoint rejects conditional writes, continuing with compare-then-put only")
                self._conditional_writes = False
                params.pop("IfNoneMatch", None)
                params.pop("IfMatch", None)
        response = self.client.put_object(**params)
        return str(response.get("ETag", "")).strip('"')

    def _sidecar(self, key: str) -> tuple[datetime | None, dict[str, Any]]:
        found = self._get(key + SIDECAR_SUFFIX)
        return _parse_sidecar(found[0] if found else None)

    def _save_sidecar(self, key: str, last_read: datetime | None, metadata: dict[str, Any]) -> None:
        self.client.put_object(
            Bucket=self.bucket,
            Key=key + SIDECAR_SUFFIX,
            Body=_dump_sidecar(last_read, metadata).encode("utf-8"),
            ContentType="application/json",
        )

    # ------------------------------------------------------------------
    # sync core
    # ------------------------------------------------------------------

    def read(self, path: str) -> MemoryFile | None:
        key = self._key(path)
        found = self._get(key)
        if found is None:
            return None
        content, etag, last_modified = found
        last_read, metadata = self._sidecar(key)
        return MemoryFile(
            path=normalize_memory_path(path),
            content=content,
            version=etag,
            description=description_from_content(content),
            last_read=last_read,
            updated_at=last_modified,
            metadata=metadata,
        )

    def write(self, path: str, content: str, expected_version: Version = None) -> MemoryFile:
        key = self._key(path)
        etag = self._put(
            key,
            content,
            expected_etag=str(expected_version) if expected_version is not None else None,
            must_not_exist=expected_version is None,
        )
        last_read, metadata = self._sidecar(key)
        return MemoryFile(
            path=normalize_memory_path(path),
            content=content,
            version=etag,
            description=description_from_content(content),
            last_read=last_read,
            updated_at=datetime.now(timezone.utc),
            metadata=metadata,
        )

    def delete(self, path: str) -> None:
        key = self._key(path)
        for k in (key, key + SIDECAR_SUFFIX):
            try:
                self.client.delete_object(Bucket=self.bucket, Key=k)
            except Exception as e:  # noqa: BLE001
                if _error_code(e) not in _NOT_FOUND:
                    raise

    def list(self, prefix: str) -> list[MemoryFile]:
        key_prefix = self._key_prefix(prefix)
        keys: list[tuple[str, datetime, str]] = []
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=key_prefix):
            for obj in page.get("Contents", []) or []:
                key = obj["Key"]
                if key.endswith(SIDECAR_SUFFIX) or not key.endswith(".md"):
                    continue
                keys.append(
                    (key, obj.get("LastModified") or datetime.now(timezone.utc), str(obj.get("ETag", "")).strip('"'))
                )
        keys.sort()

        def load(item: tuple[str, datetime, str]) -> MemoryFile | None:
            key, last_modified, etag = item
            found = self._get(key)
            if found is None:
                return None
            content, _, _ = found
            last_read, metadata = self._sidecar(key)
            try:
                path = normalize_memory_path(self._path(key))
            except Exception:  # noqa: BLE001 — foreign object under the prefix
                return None
            return MemoryFile(
                path=path,
                content=None,
                version=etag,
                description=description_from_content(content),
                last_read=last_read,
                updated_at=last_modified,
                metadata=metadata,
            )

        files = list(self._pool.map(load, keys))
        return [f for f in files if f is not None]

    def update_metadata(self, path: str, patch: dict[str, Any]) -> None:
        key = self._key(path)
        if self._head_etag(key) is None:
            return
        last_read, metadata = self._sidecar(key)
        self._save_sidecar(key, last_read, {**metadata, **patch})

    def touch_read(self, paths: list[str]) -> None:
        now = datetime.now(timezone.utc)
        for p in paths:
            key = self._key(p)
            if self._head_etag(key) is None:
                continue
            _, metadata = self._sidecar(key)
            self._save_sidecar(key, now, metadata)

    # ------------------------------------------------------------------
    # async facade
    # ------------------------------------------------------------------

    async def aread(self, path: str) -> MemoryFile | None:
        return await asyncio.to_thread(self.read, path)

    async def aread_many(self, paths: list[str]) -> list[MemoryFile]:
        files = await asyncio.gather(*(self.aread(p) for p in paths))
        return [f for f in files if f is not None]

    async def alist(self, prefix: str) -> list[MemoryFile]:
        return await asyncio.to_thread(self.list, prefix)

    async def awrite(self, path: str, content: str, expected_version: Version = None) -> MemoryFile:
        return await asyncio.to_thread(self.write, path, content, expected_version)

    async def adelete(self, path: str) -> None:
        await asyncio.to_thread(self.delete, path)

    async def atouch_read(self, paths: list[str]) -> None:
        await asyncio.to_thread(self.touch_read, paths)

    async def aupdate_metadata(self, path: str, patch: dict[str, Any]) -> None:
        await asyncio.to_thread(self.update_metadata, path, patch)
