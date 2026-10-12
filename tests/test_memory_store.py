"""Long-term memory stores: path validation, versioning, local files."""

import asyncio
import os

import pytest

from gwenflow.memory import (
    InMemoryMemoryStore,
    LocalFileMemoryStore,
    MemoryConflictError,
    MemoryPathError,
    MemoryStore,
    normalize_memory_path,
)
from gwenflow.memory.store import normalize_memory_prefix
from gwenflow.utils.markdown import parse_frontmatter, render_frontmatter

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("/memories/user/MEMORY.md", "/memories/user/MEMORY.md"),
        ("/memories/user/projects/orion.md", "/memories/user/projects/orion.md"),
        ("/memories/user/./MEMORY.md", "/memories/user/MEMORY.md"),
        ("/memories/user/a/../MEMORY.md", "/memories/user/MEMORY.md"),
        ("  /memories/thread/abc/NOTES.md ", "/memories/thread/abc/NOTES.md"),
    ],
)
def test_normalize_path_accepts_canonical(raw, expected):
    assert normalize_memory_path(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "MEMORY.md",
        "/etc/passwd.md",
        "/memories/../etc/x.md",
        "/memories/../../secrets.env",
        "/memories/user/notes.txt",
        "/memories/user/.md",
        "/memories/user/%2e%2e/x.md",
        "/memories/user\\..\\x.md",
        "/memories/user/sp ace.md",
        "/memories",
    ],
)
def test_normalize_path_rejects_traversal_and_outside_root(raw):
    with pytest.raises(MemoryPathError):
        normalize_memory_path(raw)


def test_normalize_prefix():
    assert normalize_memory_prefix("/memories/user") == "/memories/user/"
    assert normalize_memory_prefix("/memories/user/") == "/memories/user/"
    assert normalize_memory_prefix("/memories") == "/memories/"
    with pytest.raises(MemoryPathError):
        normalize_memory_prefix("/memories/../x")
    with pytest.raises(MemoryPathError):
        normalize_memory_prefix("/tmp")


# ---------------------------------------------------------------------------
# Frontmatter helpers
# ---------------------------------------------------------------------------


def test_parse_frontmatter_roundtrip():
    doc = render_frontmatter({"name": "orion", "description": "Orion project", "type": "projects"}, "# Orion\n- hi")
    meta, body = parse_frontmatter(doc)
    assert meta == {"name": "orion", "description": "Orion project", "type": "projects"}
    assert body == "# Orion\n- hi"


def test_parse_frontmatter_without_block_and_with_horizontal_rule():
    assert parse_frontmatter("# Title\n\n---\n\nbody") == ({}, "# Title\n\n---\n\nbody")
    assert parse_frontmatter("---\n- not a mapping\n---\nbody")[0] == {}


# ---------------------------------------------------------------------------
# InMemoryMemoryStore
# ---------------------------------------------------------------------------


def test_inmemory_is_a_memory_store():
    assert isinstance(InMemoryMemoryStore(), MemoryStore)


def test_inmemory_versioning_increments():
    store = InMemoryMemoryStore()
    assert store.read("/memories/user/MEMORY.md") is None
    f1 = store.write("/memories/user/MEMORY.md", "v1")
    assert f1.version == 1
    f2 = store.write("/memories/user/MEMORY.md", "v2", expected_version=1)
    assert f2.version == 2
    assert store.read("/memories/user/MEMORY.md").content == "v2"


def test_inmemory_conflict_raises():
    store = InMemoryMemoryStore()
    store.write("/memories/user/MEMORY.md", "v1")
    with pytest.raises(MemoryConflictError):
        store.write("/memories/user/MEMORY.md", "v2", expected_version=0)
    # expected_version=None means "must not exist": a second creation conflicts too
    with pytest.raises(MemoryConflictError) as exc:
        store.write("/memories/user/MEMORY.md", "v3")
    assert exc.value.actual == 1


def test_inmemory_list_and_descriptions():
    store = InMemoryMemoryStore()
    store.write("/memories/user/MEMORY.md", "# core")
    store.write(
        "/memories/user/projects/orion.md",
        render_frontmatter({"name": "orion", "description": "Orion project, phase 2"}, "- bullet"),
    )
    store.write("/memories/thread/t1/NOTES.md", "# notes")
    entries = store.list("/memories/user/")
    assert [e.path for e in entries] == ["/memories/user/MEMORY.md", "/memories/user/projects/orion.md"]
    assert entries[0].content is None
    assert entries[1].description == "Orion project, phase 2"


def test_inmemory_async_facade_and_touch_read():
    async def scenario():
        store = InMemoryMemoryStore()
        await store.awrite("/memories/user/MEMORY.md", "x")
        files = await store.aread_many(["/memories/user/MEMORY.md", "/memories/user/missing.md"])
        assert [f.path for f in files] == ["/memories/user/MEMORY.md"]
        await store.atouch_read(["/memories/user/MEMORY.md"])
        assert (await store.aread("/memories/user/MEMORY.md")).last_read is not None
        await store.adelete("/memories/user/MEMORY.md")
        assert await store.aread("/memories/user/MEMORY.md") is None

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# LocalFileMemoryStore
# ---------------------------------------------------------------------------


def test_localfile_roundtrip_and_list(tmp_path):
    store = LocalFileMemoryStore(tmp_path)
    written = store.write("/memories/user/projects/orion.md", "---\ndescription: Orion\n---\n# Orion\n")
    assert (tmp_path / "user" / "projects" / "orion.md").read_text() == "---\ndescription: Orion\n---\n# Orion\n"
    assert written.version is not None
    again = store.read("/memories/user/projects/orion.md")
    assert again.version == written.version
    assert again.description == "Orion"
    store.write("/memories/user/MEMORY.md", "core")
    assert [f.path for f in store.list("/memories/user/")] == [
        "/memories/user/MEMORY.md",
        "/memories/user/projects/orion.md",
    ]
    assert store.list("/memories/thread/") == []


def test_localfile_conflict_detection(tmp_path):
    store = LocalFileMemoryStore(tmp_path)
    first = store.write("/memories/user/MEMORY.md", "v1")
    second = store.write("/memories/user/MEMORY.md", "v2", expected_version=first.version)
    assert second.version != first.version
    with pytest.raises(MemoryConflictError):
        store.write("/memories/user/MEMORY.md", "v3", expected_version=first.version)
    with pytest.raises(MemoryConflictError):
        store.write("/memories/user/MEMORY.md", "v3")  # exists, expected None


def test_localfile_rejects_escape(tmp_path):
    store = LocalFileMemoryStore(tmp_path / "mem")
    with pytest.raises(MemoryPathError):
        store.read("/memories/../outside.md")
    outside = tmp_path / "outside"
    outside.mkdir()
    os.symlink(outside, tmp_path / "mem" / "user")
    with pytest.raises(MemoryPathError):
        store.write("/memories/user/MEMORY.md", "escaped")
    assert not (outside / "MEMORY.md").exists()
