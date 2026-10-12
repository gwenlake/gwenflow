"""S3MemoryStore against a fake boto3 client (offline), with and without conditional writes."""

# ruff: noqa: N803 — the fake client mirrors boto3's CamelCase keyword arguments

import asyncio
import hashlib
from datetime import datetime, timezone

import pytest

from gwenflow.memory import MemoryConflictError, MemoryStore, MemoryToolset, S3MemoryStore

USER = "/memories/user/MEMORY.md"
ORION = "/memories/user/projects/orion.md"


class _ClientError(Exception):
    def __init__(self, code: str):
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


class _Body:
    def __init__(self, data: bytes):
        self._data = data

    def read(self) -> bytes:
        return self._data


class FakeS3:
    """Enough of boto3's S3 client for the store: ETag = md5, optional conditional puts."""

    def __init__(self, *, supports_conditional: bool = True):
        self.objects: dict[str, bytes] = {}
        self.modified: dict[str, datetime] = {}
        self.supports_conditional = supports_conditional
        self.puts = 0

    @staticmethod
    def _etag(data: bytes) -> str:
        return '"' + hashlib.md5(data).hexdigest() + '"'

    def put_object(self, *, Bucket, Key, Body, ContentType=None, IfMatch=None, IfNoneMatch=None):
        if IfMatch is not None or IfNoneMatch is not None:
            if not self.supports_conditional:
                raise _ClientError("NotImplemented")
            current = self.objects.get(Key)
            if IfNoneMatch == "*" and current is not None:
                raise _ClientError("PreconditionFailed")
            if IfMatch is not None and (current is None or self._etag(current).strip('"') != IfMatch):
                raise _ClientError("PreconditionFailed")
        self.puts += 1
        self.objects[Key] = Body
        self.modified[Key] = datetime.now(timezone.utc)
        return {"ETag": self._etag(Body)}

    def get_object(self, *, Bucket, Key):
        if Key not in self.objects:
            raise _ClientError("NoSuchKey")
        data = self.objects[Key]
        return {"Body": _Body(data), "ETag": self._etag(data), "LastModified": self.modified[Key]}

    def head_object(self, *, Bucket, Key):
        if Key not in self.objects:
            raise _ClientError("404")
        return {"ETag": self._etag(self.objects[Key])}

    def delete_object(self, *, Bucket, Key):
        self.objects.pop(Key, None)
        self.modified.pop(Key, None)
        return {}

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        client = self

        class Paginator:
            def paginate(self, *, Bucket, Prefix):
                contents = [
                    {"Key": k, "LastModified": client.modified[k], "ETag": client._etag(v)}
                    for k, v in sorted(client.objects.items())
                    if k.startswith(Prefix)
                ]
                yield {"Contents": contents}

        return Paginator()


@pytest.fixture(params=[True, False], ids=["conditional", "compare-then-put"])
def store(request):
    return S3MemoryStore(
        "bucket",
        "memories/env1/user1",
        client=FakeS3(supports_conditional=request.param),
        conditional_writes=request.param,
    )


def test_is_a_memory_store(store):
    assert isinstance(store, MemoryStore)


def test_roundtrip_keys_versions_and_listing(store):
    assert store.read(USER) is None
    first = store.write(USER, "# User memory\n")
    assert store.client.objects["memories/env1/user1/user/MEMORY.md"] == b"# User memory\n"
    assert isinstance(first.version, str) and first.version
    second = store.write(USER, "# User memory\n- x\n", expected_version=first.version)
    assert second.version != first.version
    store.write(ORION, "---\ndescription: Orion project\n---\n- email\n")
    index = store.list("/memories/user/")
    assert [(f.path, f.description, f.content) for f in index] == [
        (USER, None, None),
        (ORION, "Orion project", None),
    ]
    assert all(f.updated_at is not None for f in index)
    assert store.read(USER).content == "# User memory\n- x\n"


def test_conflicts(store):
    first = store.write(USER, "v1")
    with pytest.raises(MemoryConflictError) as exc:
        store.write(USER, "v2")  # must not exist, but it does
    assert exc.value.actual == first.version
    with pytest.raises(MemoryConflictError):
        store.write(USER, "v2", expected_version="stale-etag")
    assert store.read(USER).content == "v1"


def test_metadata_and_last_read_live_in_the_sidecar(store):
    store.write(ORION, "- x")
    assert store.read(ORION).last_read is None and store.read(ORION).metadata == {}
    store.update_metadata(ORION, {"entities": [{"type": "project", "id": "42"}]})
    store.update_metadata(ORION, {"curated_at": "2026-10-12"})
    store.touch_read([ORION, "/memories/user/missing.md"])
    file = store.read(ORION)
    assert file.metadata == {"entities": [{"type": "project", "id": "42"}], "curated_at": "2026-10-12"}
    assert file.last_read is not None
    assert "memories/env1/user1/user/projects/orion.md.meta.json" in store.client.objects
    # the sidecar survives a content rewrite and vanishes with the file
    store.write(ORION, "- y", expected_version=file.version)
    assert store.read(ORION).metadata["curated_at"] == "2026-10-12"
    store.delete(ORION)
    assert store.read(ORION) is None
    assert not any(k.endswith("orion.md.meta.json") for k in store.client.objects)


def test_unsupported_conditional_headers_fall_back_and_are_remembered():
    client = FakeS3(supports_conditional=False)
    store = S3MemoryStore("bucket", "m", client=client, conditional_writes=True)
    store.write(USER, "v1")
    assert store._conditional_writes is False
    store.write(USER, "v2", expected_version=store.read(USER).version)
    assert store.read(USER).content == "v2"


def test_conflict_is_caught_even_when_the_server_ignores_preconditions():
    """Garage-style server: accepts If-Match and ignores it; compare-then-put must still catch it."""

    class IgnoringS3(FakeS3):
        def put_object(self, *, Bucket, Key, Body, ContentType=None, IfMatch=None, IfNoneMatch=None):
            return super().put_object(Bucket=Bucket, Key=Key, Body=Body, ContentType=ContentType)

    store = S3MemoryStore("bucket", "m", client=IgnoringS3(), conditional_writes=True)
    first = store.write(USER, "v1")
    store.write(USER, "v2", expected_version=first.version)
    with pytest.raises(MemoryConflictError):
        store.write(USER, "v3", expected_version=first.version)
    assert store.read(USER).content == "v2"


def test_toolset_over_s3(store):
    async def scenario():
        toolset = MemoryToolset(store, always=[USER], browse=["/memories/user/"])
        await toolset.aload()
        tools = {t.name: t for t in toolset.get_tools()}
        assert tools["write_topic_file"].run(path=ORION, description="Orion", content="- email").startswith("OK.")
        assert tools["memory_edit"].run(path=USER, old_str="## Notes\n", new_str="## Notes\n- a\n").startswith("OK.")
        report = await toolset.aflush()
        assert report.ok and {f.path for f in report.written} == {USER, ORION}
        again = MemoryToolset(store, always=[USER], browse=["/memories/user/"])
        await again.aload()
        assert f"- {ORION} — Orion" in again.get_instructions()
        assert "- email" in await asyncio.to_thread(
            {t.name: t for t in again.get_tools()}["read_memory"].run, path=ORION
        )

    asyncio.run(scenario())
