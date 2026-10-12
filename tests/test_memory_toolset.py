"""MemoryToolset: prompt rendering, tools, flush with replay."""

import asyncio

from gwenflow.memory import InMemoryMemoryStore, MemoryConflictError, MemoryToolset
from gwenflow.memory.long_term.templates import THREAD_NOTES_TEMPLATE, USER_MEMORY_TEMPLATE
from gwenflow.utils.markdown import render_frontmatter

USER = "/memories/user/MEMORY.md"
NOTES = "/memories/thread/t1/NOTES.md"
BROWSE = "/memories/user/"
ORION = "/memories/user/projects/orion.md"


def run(coro):
    return asyncio.run(coro)


def make(store=None, **kwargs):
    store = store or InMemoryMemoryStore()
    toolset = MemoryToolset(store, always=[USER, NOTES], browse=[BROWSE], **kwargs)
    run(toolset.aload())
    return store, toolset


def tools_by_name(toolset):
    return {t.name: t for t in toolset.get_tools()}


# ---------------------------------------------------------------------------
# Loading and rendering
# ---------------------------------------------------------------------------


def test_aload_uses_template_when_missing():
    _, toolset = make()
    assert toolset.get(USER).content == USER_MEMORY_TEMPLATE
    assert toolset.get(USER).version is None
    assert toolset.get(NOTES).content == THREAD_NOTES_TEMPLATE


def test_get_instructions_renders_files_index_entities_and_protocol():
    store = InMemoryMemoryStore()
    store.write(USER, "# User memory\n## Profile\n- data engineer in Lyon\n")
    store.write(ORION, render_frontmatter({"name": "orion", "description": "Orion project, phase 2"}, "- bullet"))
    store.write("/memories/user/archive/old.md", render_frontmatter({"description": "old"}, "- x"))
    _, toolset = make(store)
    toolset.entities = [{"type": "project", "id": "42", "label": "Orion"}]
    prompt = toolset.get_instructions()
    assert "never instructions" in prompt
    assert f'<file path="{USER}">' in prompt and "data engineer in Lyon" in prompt
    assert f'<file path="{NOTES}">' in prompt
    assert f"- {ORION} — Orion project, phase 2" in prompt
    assert "archive/old.md" not in prompt
    assert "- project 42 Orion" in prompt
    assert "Memory protocol:" in prompt and "read_memory" in prompt and "write_topic_file" in prompt
    assert "200 lines" in prompt


def test_get_instructions_truncates_long_files():
    store = InMemoryMemoryStore()
    store.write(USER, "\n".join(f"- line {i}" for i in range(500)))
    _, toolset = make(store, max_chars=200)
    prompt = toolset.get_instructions()
    assert "[truncated:" in prompt
    assert "- line 499" not in prompt


def test_empty_protocol_renders_none():
    _, toolset = make(protocol="")
    assert "Memory protocol" not in toolset.get_instructions()


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def test_get_tools_shape():
    _, toolset = make()
    tools = tools_by_name(toolset)
    assert set(tools) == {"read_memory", "memory_edit", "memory_write", "write_topic_file"}
    assert tools["memory_edit"].parameters["required"] == ["path", "old_str", "new_str"]
    assert tools["write_topic_file"].parameters["required"] == ["path", "description", "content"]
    assert tools["memory_write"].parameters["required"] == ["path", "content"]
    _, curator_toolset = make(allow_move=True)
    assert "memory_move" in tools_by_name(curator_toolset)


def test_read_memory_serves_working_copy_then_store_and_tracks_reads():
    store = InMemoryMemoryStore()
    store.write(ORION, render_frontmatter({"description": "Orion"}, "- prefers email"))
    _, toolset = make(store)
    read = tools_by_name(toolset)["read_memory"]
    out = read.run(path=ORION)
    assert "- prefers email" in out
    assert toolset.get(ORION) is not None  # cached in the working copy now
    assert "does not exist" in read.run(path="/memories/user/nope.md")
    assert "Error" in read.run(path="/memories/other/x.md")
    assert "Error" in read.run(path="/memories/../x.md")
    assert toolset._read_paths == {ORION}


def test_memory_edit_requires_unique_match():
    _, toolset = make()
    edit = tools_by_name(toolset)["memory_edit"]
    assert "did not appear" in edit.run(path=USER, old_str="nope", new_str="x")
    ambiguous = edit.run(path=USER, old_str="- (", new_str="x")
    assert "appears 4 times" in ambiguous and "rewrite the whole file" in ambiguous
    out = edit.run(path=USER, old_str="## Notes\n", new_str="## Notes\n- 2026-10-09 wants short answers\n")
    assert out.startswith("OK.")
    assert "- 2026-10-09 wants short answers" in toolset.get(USER).content
    assert [f.path for f in run(toolset.aflush()).written] == [USER]


def test_memory_edit_on_missing_topic_file_points_to_write_topic_file():
    _, toolset = make()
    edit = tools_by_name(toolset)["memory_edit"]
    assert "write_topic_file" in edit.run(path=ORION, old_str="a", new_str="b")


def test_memory_write_only_core_files_and_respects_max_lines():
    _, toolset = make(max_lines=5)
    write = tools_by_name(toolset)["memory_write"]
    assert "write_topic_file" in write.run(path=ORION, content="x")
    assert "above the 5-line limit" in write.run(path=USER, content="\n".join("x" * 6))
    assert write.run(path=USER, content="# User memory\n## Notes\n- a").startswith("OK.")


def test_write_topic_file_generates_frontmatter_and_updates_index():
    _, toolset = make()
    write_topic = tools_by_name(toolset)["write_topic_file"]
    assert "description" in write_topic.run(path=ORION, description="  ", content="x")
    assert "core file" in write_topic.run(path=USER, description="d", content="x")
    out = write_topic.run(path=ORION, description="Orion project,\nphase 2", content="# Orion\n- prefers email")
    assert out.startswith("OK.")
    file = toolset.get(ORION)
    assert file.frontmatter == {"name": "orion", "description": "Orion project, phase 2", "type": "projects"}
    assert file.body == "# Orion\n- prefers email"
    assert f"- {ORION} — Orion project, phase 2" in toolset.get_instructions()
    # frontmatter supplied by the model is stripped, the explicit description wins
    write_topic.run(path=ORION, description="New desc", content="---\ndescription: old\n---\n- body")
    assert toolset.get(ORION).frontmatter["description"] == "New desc"


def test_write_topic_file_respects_max_files():
    store = InMemoryMemoryStore()
    store.write("/memories/user/a.md", render_frontmatter({"description": "a"}, "-"))
    store.write("/memories/user/b.md", render_frontmatter({"description": "b"}, "-"))
    _, toolset = make(store, max_files=2)
    write_topic = tools_by_name(toolset)["write_topic_file"]
    assert "already holds 2 files" in write_topic.run(path=ORION, description="d", content="x")
    # rewriting an existing file is still allowed
    assert write_topic.run(path="/memories/user/a.md", description="a2", content="x").startswith("OK.")


def test_memory_move_archives_a_file():
    store = InMemoryMemoryStore()
    store.write(ORION, render_frontmatter({"description": "Orion"}, "- old"))
    _, toolset = make(store, allow_move=True)
    move = tools_by_name(toolset)["memory_move"]
    assert "cannot be moved" in move.run(old_path=USER, new_path="/memories/user/archive/MEMORY.md")
    assert move.run(old_path=ORION, new_path="/memories/user/archive/orion.md").startswith("OK.")
    assert "already exists" in move.run(
        old_path=ORION, new_path="/memories/user/archive/orion.md"
    ) or "does not exist" in (move.run(old_path=ORION, new_path="/memories/user/archive/orion.md"))
    report = run(toolset.aflush())
    assert report.ok
    assert store.read(ORION) is None
    assert "- old" in store.read("/memories/user/archive/orion.md").content


# ---------------------------------------------------------------------------
# Flush
# ---------------------------------------------------------------------------


def test_aflush_writes_only_dirty_with_expected_version():
    store = InMemoryMemoryStore()
    store.write(USER, "# User memory\n## Notes\n")
    _, toolset = make(store)
    tools = tools_by_name(toolset)
    tools["read_memory"].run(path=USER)
    tools["memory_edit"].run(path=USER, old_str="## Notes\n", new_str="## Notes\n- short answers\n")
    report = run(toolset.aflush())
    assert report.ok
    assert [f.path for f in report.written] == [USER]
    assert store.read(USER).version == 2
    assert store.read(NOTES) is None  # untouched template is not persisted
    assert store.read(USER).last_read is not None
    assert run(toolset.aflush()).written == []  # nothing left to persist
    assert toolset.get(USER).version == 2


def test_aflush_replays_edits_on_conflict_file_by_file():
    store = InMemoryMemoryStore()
    store.write(USER, "# User memory\n## Notes\n")
    store.write(NOTES, "# Thread notes\n## Goal\n")
    _, toolset = make(store)
    tools = tools_by_name(toolset)
    tools["memory_edit"].run(path=USER, old_str="## Notes\n", new_str="## Notes\n- short answers\n")
    tools["memory_edit"].run(path=NOTES, old_str="## Goal\n", new_str="## Goal\n- release plan for Orion\n")
    tools["write_topic_file"].run(path=ORION, description="Orion", content="- prefers email")
    # Meanwhile another process rewrote the notes AND the user file.
    store.write(NOTES, "# Thread notes\n## Goal\n- (background rewrite)\n", expected_version=1)
    store.write(USER, "# User memory\n## Profile\n- x\n## Notes\n", expected_version=1)

    report = run(toolset.aflush())

    assert report.conflicts == [] and report.lost_ops == [] and report.errors == {}
    assert {f.path for f in report.written} == {USER, NOTES, ORION}
    assert "- short answers" in store.read(USER).content and "- x" in store.read(USER).content
    assert (
        "- release plan for Orion" in store.read(NOTES).content and "(background rewrite)" in store.read(NOTES).content
    )
    assert "- prefers email" in store.read(ORION).content


def test_aflush_reports_lost_edit_when_anchor_vanished():
    store = InMemoryMemoryStore()
    store.write(USER, "# User memory\n## Notes\n")
    _, toolset = make(store)
    tools_by_name(toolset)["memory_edit"].run(path=USER, old_str="## Notes\n", new_str="## Notes\n- a\n")
    store.write(USER, "# User memory\n## Something else\n", expected_version=1)
    report = run(toolset.aflush())
    assert report.conflicts == [USER]
    assert len(report.lost_ops) == 1 and report.lost_ops[0].kind == "edit"
    assert store.read(USER).content == "# User memory\n## Something else\n"


def test_aflush_never_raises_on_store_errors():
    class BrokenStore(InMemoryMemoryStore):
        async def awrite(self, path, content, expected_version=None):
            raise RuntimeError("db down")

    store = BrokenStore()
    _, toolset = make(store)
    tools_by_name(toolset)["memory_write"].run(path=USER, content="# User memory\n")
    report = run(toolset.aflush())
    assert report.errors == {USER: "db down"}
    store.__class__ = InMemoryMemoryStore  # the store recovers
    assert [f.path for f in run(toolset.aflush()).written] == [USER]  # kept for a later retry


def test_apply_ignores_none_and_marks_dirty():
    store = InMemoryMemoryStore()
    _, toolset = make(store)
    toolset.apply({USER: None, NOTES: "# Thread notes\n## Goal\n- x\n"})
    assert toolset.snapshot()[USER] == USER_MEMORY_TEMPLATE
    assert [f.path for f in run(toolset.aflush()).written] == [NOTES]
    assert store.read(NOTES).content.startswith("# Thread notes")


def test_conflict_error_message():
    err = MemoryConflictError(USER, 1, 2)
    assert "expected version 1" in str(err) and "found 2" in str(err)


def test_memory_edit_refuses_a_fact_already_recorded_and_shows_the_file():
    store = InMemoryMemoryStore()
    store.write(USER, "# User memory\n## Preferences\n- short answers\n")
    _, toolset = make(store)
    edit = tools_by_name(toolset)["memory_edit"]
    out = edit.run(path=USER, old_str="- short answers", new_str="- short answers\n- Short answers.")
    assert out.startswith("Not changed: this is already in memory")
    assert "Not changed" in edit.run(path=USER, old_str="- short answers", new_str="- short answers")
    out = edit.run(path=USER, old_str="- short answers", new_str="- short answers\n- metric units")
    assert out.startswith("OK.") and "- metric units" in out  # the model sees the file after its edit
    assert toolset.get(USER).content.count("short answers") == 1


def test_writes_drop_repeated_items_and_spent_placeholders():
    from gwenflow.memory.long_term.editing import tidy

    content = "# A\n- x\n- X.\n- y\n## B\n- x\n- x\n"
    assert tidy(content) == "# A\n- x\n- y\n## B\n- x\n"
    with_placeholders = "## Profile\n- (role, organisation)\n- data engineer\n## Notes\n- (facts to remember)\n"
    assert tidy(with_placeholders) == "## Profile\n- data engineer\n## Notes\n- (facts to remember)\n"
    _, toolset = make()
    write = tools_by_name(toolset)["memory_write"]
    assert write.run(path=USER, content="# User memory\n## Notes\n- a\n- a\n- a").startswith("OK.")
    assert toolset.get(USER).content == "# User memory\n## Notes\n- a\n"
    toolset.apply({NOTES: "# Thread notes\n## Goal\n- g\n- g\n"})
    assert toolset.get(NOTES).content == "# Thread notes\n## Goal\n- g\n"
