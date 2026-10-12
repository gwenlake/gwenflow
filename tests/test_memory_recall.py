"""Automatic recall, tool selection and topic-file creation by the consolidator."""

import asyncio

import pytest

from gwenflow.memory import InMemoryMemoryStore, MemoryConsolidator, MemoryFile, MemoryToolset
from gwenflow.memory.long_term.recall import select_relevant
from gwenflow.types import Message, ModelResponse, RequestUsage, TextContent
from gwenflow.utils.markdown import render_frontmatter

USER = "/memories/user/MEMORY.md"
NOTES = "/memories/thread/t1/NOTES.md"
ORION = "/memories/user/projects/orion-launch.md"
VEGA = "/memories/user/projects/vega-migration.md"
REVIEW = "/memories/user/topics/quarterly-review.md"


def entry(path, description):
    return MemoryFile(path=path, description=description)


INDEX = [
    entry(ORION, "Orion launch (ref 123), owned by Alice, mobile release"),
    entry(VEGA, "Vega migration (ref 456), database move"),
    entry(REVIEW, "Quarterly review of open tickets, Q4 2026"),
    entry("/memories/user/archive/old.md", "Old pilot, closed"),
]


# ---------------------------------------------------------------------------
# select_relevant
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Prepare my meeting about Orion next week", [ORION]),
        ("Where is the ORION LAUNCH at?", [ORION]),
        ("ticket 123 was reopened", [ORION]),
        ("compare Orion and the Vega migration", [ORION, VEGA]),
        ("is the quarterly review ready?", [REVIEW]),
        ("What day is it?", []),
        ("ask Alice", []),  # one weak description word is not enough
        ("prepare a mobile app", []),
        ("", []),
    ],
)
def test_select_relevant(text, expected):
    assert sorted(h.path for h in select_relevant(INDEX, text)) == sorted(expected)


def test_select_relevant_ignores_accents_case_and_respects_limit():
    hits = select_relevant(INDEX, "ORIÓN, vega, quarterly", limit=2)
    assert len(hits) == 2
    assert select_relevant(INDEX, "orion", limit=0) == []


def test_years_are_not_identifiers():
    assert select_relevant([entry(REVIEW, "Review 2026")], "the 2026 budget") == []


def test_application_stopwords_silence_domain_words():
    index = [entry("/memories/user/projects/northwind-portal.md", "Northwind portal (ref 77)")]
    assert select_relevant(index, "the northwind dashboard") != []
    assert select_relevant(index, "the northwind dashboard", stopwords=["Northwind"]) == []


# ---------------------------------------------------------------------------
# recall through the toolset
# ---------------------------------------------------------------------------


def seeded_store():
    store = InMemoryMemoryStore()
    store.write(USER, "# User memory\n## Notes\n")
    store.write(ORION, render_frontmatter({"description": INDEX[0].description}, "- owner Alice"))
    store.write(VEGA, render_frontmatter({"description": INDEX[1].description}, "- cutover on a Sunday"))
    return store


def test_aload_with_query_recalls_and_renders_matching_files():
    store = seeded_store()
    toolset = MemoryToolset(store, always=[USER, NOTES], browse=["/memories/user/"])
    asyncio.run(toolset.aload(query="Prepare my meeting about Orion"))
    assert toolset.recalled == [ORION]
    prompt = toolset.get_instructions()
    assert f'<file path="{ORION}" recalled="true">' in prompt and "- owner Alice" in prompt
    assert "- cutover on a Sunday" not in prompt  # not recalled: index line only
    assert f"- {ORION} — Orion launch (ref 123), owned by Alice, mobile release (shown above)" in prompt
    # recalled files count as read and are editable without a read_memory round trip
    asyncio.run(toolset.aflush())
    assert store.read(ORION).last_read is not None and store.read(VEGA).last_read is None


def test_aload_without_query_or_with_recall_disabled():
    store = seeded_store()
    toolset = MemoryToolset(store, always=[USER], browse=["/memories/user/"], recall_limit=0)
    asyncio.run(toolset.aload(query="Orion"))
    assert toolset.recalled == []
    toolset = MemoryToolset(store, always=[USER], browse=["/memories/user/"])
    asyncio.run(toolset.aload())
    assert toolset.recalled == []


def test_recall_stopwords_reach_the_toolset():
    store = seeded_store()
    toolset = MemoryToolset(store, always=[USER], browse=["/memories/user/"], recall_stopwords=["orion", "launch"])
    asyncio.run(toolset.aload(query="Orion launch"))
    assert toolset.recalled == []


# ---------------------------------------------------------------------------
# tool selection and protocol
# ---------------------------------------------------------------------------


def test_conversational_tool_subset_and_protocol():
    toolset = MemoryToolset(
        InMemoryMemoryStore(), always=[USER, NOTES], browse=["/memories/user/"], tools=("read_memory", "memory_edit")
    )
    asyncio.run(toolset.aload())
    assert [t.name for t in toolset.get_tools()] == ["read_memory", "memory_edit"]
    prompt = toolset.get_instructions()
    assert "recorded automatically after your answer" in prompt
    assert "write_topic_file" not in prompt
    assert "read_memory" in prompt


def test_no_write_tools_protocol():
    toolset = MemoryToolset(InMemoryMemoryStore(), always=[USER], browse=["/memories/user/"], tools=("read_memory",))
    asyncio.run(toolset.aload())
    assert "Do not write memory" in toolset.get_instructions()


def test_default_tools_unchanged_and_unknown_rejected():
    toolset = MemoryToolset(InMemoryMemoryStore(), always=[USER], browse=["/memories/user/"], allow_move=True)
    assert [t.name for t in toolset.get_tools()] == [
        "read_memory",
        "memory_edit",
        "memory_write",
        "write_topic_file",
        "memory_move",
    ]
    with pytest.raises(ValueError):
        MemoryToolset(InMemoryMemoryStore(), always=[USER], tools=("delete_everything",))


# ---------------------------------------------------------------------------
# consolidator: topic files
# ---------------------------------------------------------------------------


class Canned:
    def __init__(self, text):
        self.text = text
        self.inputs = []

    def invoke(self, input):
        self.inputs.append(input)
        return ModelResponse(
            parts=[TextContent(content=self.text)], usage=RequestUsage(input_tokens=1, output_tokens=1)
        )


TURN = [
    Message(role="user", content="Orion: Bob takes over from Alice next month"),
    Message(role="assistant", content="Noted."),
]


def consolidator(text, **kw):
    return MemoryConsolidator(
        llm=Canned(text), creatable_prefixes=["/memories/user/projects/", "/memories/user/topics/"], **kw
    )


def test_updates_a_recalled_topic_file_and_keeps_its_frontmatter():
    orion = render_frontmatter(
        {"name": "orion-launch", "description": "Orion launch (ref 123)", "type": "projects"}, "- owner Alice"
    )
    answer = f'<file path="{NOTES}">\n# notes\n</file>\n<file path="{ORION}">\n- owner Bob (from next month)\n</file>'
    result = consolidator(answer).consolidate({NOTES: "# old", ORION: orion}, TURN)
    assert result.files[ORION].startswith("---\nname: orion-launch\ndescription: Orion launch (ref 123)")
    assert "- owner Bob (from next month)" in result.files[ORION]


def test_creates_new_topic_files_under_allowed_prefixes_only():
    answer = (
        f'<file path="{NOTES}" unchanged="true"></file>\n'
        f'<file path="{VEGA}">\n---\ndescription: Vega migration (ref 456)\n---\n- cutover on a Sunday\n</file>\n'
        '<file path="/memories/user/MEMORY.md">\nhijack\n</file>\n'
        '<file path="/memories/user/archive/x.md">\n---\ndescription: x\n---\n- x\n</file>\n'
        '<file path="/memories/user/projects/nodesc.md">\n- body only\n</file>\n'
        '<file path="/memories/user/projects/../../escape.md">\n---\ndescription: x\n---\n- x\n</file>'
    )
    c = consolidator(answer)
    result = c.consolidate({NOTES: "# notes"}, TURN, known_paths=[USER])
    assert set(result.changed) == {VEGA}
    assert result.files[VEGA].startswith(
        "---\nname: vega-migration\ndescription: Vega migration (ref 456)\ntype: projects"
    )
    system = c.llm.inputs[0][0]["content"]
    assert "New topic files" in system and "`/memories/user/projects/`" in system


def test_never_overwrites_an_existing_file_it_was_not_given():
    answer = f'<file path="{ORION}">\n---\ndescription: Orion\n---\n- overwrite\n</file>'
    result = consolidator(answer).consolidate({NOTES: "# notes"}, TURN, known_paths=[ORION])
    assert result.changed == {}


def test_caps_new_files_and_app_instructions_reach_the_prompt():
    blocks = "\n".join(
        f'<file path="/memories/user/topics/t{i}.md">\n---\ndescription: T{i}\n---\n- x\n</file>' for i in range(4)
    )
    c = consolidator(blocks, max_new_files=2, instructions="File names are lowercase with hyphens.")
    result = c.consolidate({NOTES: "# notes"}, TURN)
    assert len(result.changed) == 2
    assert "Application rules:\nFile names are lowercase with hyphens." in c.llm.inputs[0][0]["content"]


def test_without_prefixes_nothing_can_be_created():
    answer = f'<file path="{VEGA}">\n---\ndescription: V\n---\n- x\n</file>'
    result = MemoryConsolidator(llm=Canned(answer)).consolidate({NOTES: "# notes"}, TURN)
    assert result.changed == {}
    assert "New topic files" not in MemoryConsolidator(llm=Canned(""))._system_prompt()


def test_identical_rewrite_counts_as_unchanged():
    result = consolidator(f'<file path="{NOTES}">\n# notes\n</file>').consolidate({NOTES: "# notes\n"}, TURN)
    assert result.files[NOTES] is None
