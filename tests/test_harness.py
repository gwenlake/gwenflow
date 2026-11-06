"""Harness: an agent described by a directory of files with fixed names (gwenflow.harness)."""

import textwrap
from pathlib import Path

import pytest

from gwenflow import Agent, ChatGwenlake, ChatOpenAI, Harness
from gwenflow.harness import HarnessError


def _write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text).lstrip())


TOOL_FILE = '''
def word_count(text: str) -> int:
    """Count the words of a text.

    Args:
        text: The text.
    """
    return len(text.split())


def _private(text: str) -> int:
    """Never exposed."""
    return 0


def undocumented(text: str) -> int:
    return 0
'''


@pytest.fixture
def harness_dir(tmp_path: Path) -> Path:
    root = tmp_path / "question_generation"
    _write(
        root,
        "harness.yaml",
        """
        name: question_generation
        version: 1
        description: Writes exam questions.
        model: google/gemma-4-26b-a4b-it
        model_settings:
          temperature: 0.2
          reasoning_effort: none
        max_turns: 8
        metadata:
          max_examples: 2
        """,
    )
    _write(root, "system_prompt.md", "You write questions. Today: {{ date }}.\n<!-- note for editors -->")
    _write(root, "guardrails.md", "Never give the answer in the stem.")
    _write(root, "tool_guidance.md", "Use word_count on long stems.")
    _write(
        root,
        "skills/distractor-design/SKILL.md",
        """
        ---
        name: distractor-design
        description: Plausible wrong options.
        ---
        Each distractor is a real misconception.
        """,
    )
    _write(root, "tools/text.py", TOOL_FILE)
    _write(
        root,
        "subagents/reviewer/AGENT.md",
        """
        ---
        mode: subagent
        description: Checks a question for factual errors.
        model: openai/gpt-6-luna
        model_settings:
          temperature: 0
        max_turns: 3
        tools: [word_count]
        ---
        You review medical exam questions.
        """,
    )
    _write(root, "RUN.md", "Owned by the application, ignored by gwenflow.")
    return root


@pytest.fixture(autouse=True)
def _gwenlake_key(monkeypatch):
    monkeypatch.setenv("GWENLAKE_API_KEY", "sk-test")


def _connection() -> ChatOpenAI:
    return ChatOpenAI(model="unused", api_key="sk-conn", base_url="https://gateway.example/v1")


# ---------------------------------------------------------------------- load


def test_load_reads_the_fixed_files(harness_dir):
    h = Harness.load(harness_dir)

    assert h.name == "question_generation"
    assert h.config.version == "1"
    assert h.config.metadata == {"max_examples": 2}
    assert list(h.prose) == ["system_prompt.md", "guardrails.md", "tool_guidance.md"]
    assert "note for editors" not in h.prose["system_prompt.md"]
    assert [s.name for s in h.skills] == ["distractor-design"]
    assert [t.name for t in h.tools] == ["word_count"]
    assert [s.name for s in h.subagents] == ["reviewer"]


def test_load_accepts_the_config_file_path(harness_dir):
    assert Harness.load(harness_dir / "harness.yaml").path == harness_dir


def test_an_empty_directory_is_a_harness(tmp_path):
    h = Harness.load(tmp_path)
    assert h.name == tmp_path.name
    assert h.system_prompt() == ""


@pytest.mark.parametrize(
    "config, message",
    [
        ("modle: x\n", "modle"),
        ("skills: sometimes\n", "skills"),
        ("[a, b]\n", "expected a mapping"),
        ("name: [\n", "invalid YAML"),
    ],
)
def test_a_bad_config_is_refused(tmp_path, config, message):
    (tmp_path / "harness.yaml").write_text(config)
    with pytest.raises(HarnessError, match=message):
        Harness.load(tmp_path)


def test_a_broken_tool_file_is_refused(tmp_path):
    _write(tmp_path, "tools/broken.py", "import does_not_exist\n")
    with pytest.raises(HarnessError, match="broken.py"):
        Harness.load(tmp_path)


def test_a_subagent_needs_a_description(tmp_path):
    _write(tmp_path, "subagents/x/AGENT.md", "---\nmodel: m\n---\nBody.")
    with pytest.raises(HarnessError, match="description"):
        Harness.load(tmp_path)


def test_a_subagent_only_names_known_tools(tmp_path):
    _write(tmp_path, "subagents/x/AGENT.md", "---\ndescription: d\ntools: [search]\n---\nBody.")
    with pytest.raises(HarnessError, match="unknown tool .search."):
        Harness.load(tmp_path)


# ---------------------------------------------------------------------- system prompt


def test_system_prompt_follows_the_fixed_order(harness_dir):
    prompt = Harness.load(harness_dir).system_prompt()

    assert prompt.index("You write questions") < prompt.index("Never give") < prompt.index("Use word_count")
    assert "{{" not in prompt
    assert "<name>distractor-design</name>" in prompt  # skills index, loaded on demand


def test_variables_fill_the_prose(tmp_path):
    _write(tmp_path, "system_prompt.md", "Level {{ level }}.")
    h = Harness.load(tmp_path)

    assert h.system_prompt({"level": 3}) == "Level 3."
    with pytest.raises(HarnessError, match="level"):
        h.system_prompt()


def test_inline_skills_go_into_the_prompt_without_tools(harness_dir):
    config = harness_dir / "harness.yaml"
    config.write_text(config.read_text() + "skills: inline\n")
    agent = Agent.from_harness(harness_dir, llm=_connection())

    assert "Each distractor is a real misconception." in agent.system_prompt
    assert agent.skills == []
    assert "load_skill" not in {t.name for t in agent.tools}


# ---------------------------------------------------------------------- agent


def test_build_agent_from_the_files(harness_dir):
    agent = Agent.from_harness(harness_dir)

    assert isinstance(agent.llm, ChatGwenlake)
    assert agent.llm.model == "google/gemma-4-26b-a4b-it"
    assert agent.llm.temperature == 0.2
    assert agent.llm.reasoning_effort == "none"
    assert agent.name == "question_generation"
    assert agent.description == "Writes exam questions."
    assert agent.max_turns == 8
    assert [s.name for s in agent.skills] == ["distractor-design"]
    assert {"word_count", "load_skill", "ask_reviewer"} <= {t.name for t in agent.tools}


def test_the_file_model_runs_on_the_connection_given_in_code(harness_dir):
    llm = _connection()
    agent = Agent.from_harness(harness_dir, llm=llm)

    assert isinstance(agent.llm, ChatOpenAI)
    assert agent.llm.base_url == "https://gateway.example/v1"
    assert agent.llm.model == "google/gemma-4-26b-a4b-it"
    assert llm.model == "unused" and llm.temperature is None  # the caller's object is not touched


def test_code_wins_over_the_files(harness_dir):
    agent = Agent.from_harness(
        harness_dir,
        llm=_connection(),
        model="other-model",
        model_settings={"temperature": 0.9},
        name="renamed",
        max_turns=2,
    )

    assert agent.llm.model == "other-model"
    assert agent.llm.temperature == 0.9
    assert agent.llm.reasoning_effort == "none"  # file settings not overridden stay
    assert agent.name == "renamed"
    assert agent.max_turns == 2


def test_tools_given_in_code_are_added(harness_dir):
    def lookup(term: str) -> str:
        """Look a term up.

        Args:
            term: The term.
        """
        return term

    from gwenflow import Tool

    agent = Agent.from_harness(harness_dir, llm=_connection(), tools=[Tool(lookup)])
    assert {"word_count", "lookup"} <= {t.name for t in agent.tools}


def test_an_unknown_model_setting_is_refused(harness_dir):
    with pytest.raises(HarnessError, match="temprature"):
        Agent.from_harness(harness_dir, llm=_connection(), model_settings={"temprature": 1})


def test_a_model_is_required(tmp_path):
    with pytest.raises(HarnessError, match="no model"):
        Agent.from_harness(tmp_path)
    assert Agent.from_harness(tmp_path, llm=_connection()).llm.model == "unused"


def test_subagents_become_the_team(harness_dir):
    agent = Agent.from_harness(harness_dir, llm=_connection())
    (reviewer,) = agent.team

    assert reviewer.name == "reviewer"
    assert reviewer.description == "Checks a question for factual errors."
    assert reviewer.system_prompt == "You review medical exam questions."
    assert reviewer.max_turns == 3
    assert [t.name for t in reviewer.tools] == ["word_count"]
    # its own model and settings, on the orchestrator's connection
    assert reviewer.llm.model == "openai/gpt-6-luna"
    assert reviewer.llm.base_url == "https://gateway.example/v1"
    assert reviewer.llm.temperature == 0
    assert reviewer.llm.reasoning_effort == "none"


def test_each_build_is_a_new_agent(harness_dir):
    h = Harness.load(harness_dir)
    a, b = h.build_agent(llm=_connection()), h.build_agent(llm=_connection())
    assert a is not b and a.llm is not b.llm and a.team[0] is not b.team[0]


# ---------------------------------------------------------------------- built-in tools, approval


def test_builtin_tools_by_class_name(tmp_path):
    _write(tmp_path, "harness.yaml", "tools:\n  - ClinicalTrialsTool\n  - PubMedTool: {max_results: 5}\n")
    h = Harness.load(tmp_path)

    assert [t.name for t in h.tools] == ["ClinicalTrialsTool", "PubMedTool"]
    assert h.tools[1].max_results == 5


@pytest.mark.parametrize("name", ["NoSuchTool", "Tool", "MCPServerSse"])
def test_only_ready_made_tools_are_builtin(tmp_path, name):
    _write(tmp_path, "harness.yaml", f"tools: [{name}]\n")
    with pytest.raises(HarnessError, match="unknown tool"):
        Harness.load(tmp_path)


def test_approval_marks_file_and_builtin_tools(harness_dir):
    config = harness_dir / "harness.yaml"
    config.write_text(config.read_text() + "tools: [ShellTool]\napproval: [ShellTool, word_count]\n")
    h = Harness.load(harness_dir)

    assert {t.name: t.requires_approval for t in h.tools} == {"word_count": True, "ShellTool": True}


def test_approval_names_known_tools(tmp_path):
    _write(tmp_path, "harness.yaml", "approval: [ShellTool]\n")
    with pytest.raises(HarnessError, match="approval"):
        Harness.load(tmp_path)


def test_a_subagent_may_use_a_builtin_tool(tmp_path):
    _write(tmp_path, "subagents/x/AGENT.md", "---\ndescription: d\ntools: [PubMedTool]\n---\nBody.")
    (sub,) = Harness.load(tmp_path).subagents
    assert [t.name for t in sub.tools] == ["PubMedTool"]


# ---------------------------------------------------------------------- MCP


@pytest.fixture
def _no_mcp_connection(monkeypatch):
    from gwenflow.tools.mcp import MCPServerSse, MCPServerStdio

    monkeypatch.setattr(MCPServerSse, "get_tools", lambda self: [])
    monkeypatch.setattr(MCPServerStdio, "get_tools", lambda self: [])


def test_mcp_servers_from_the_settings(tmp_path, monkeypatch, _no_mcp_connection):
    from gwenflow.tools.mcp import MCPServerSse, MCPServerStdio

    _write(
        tmp_path,
        "harness.yaml",
        """
        mcp_servers:
          - name: docs
            url: https://mcp.example/sse
            headers: {Authorization: "Bearer ${DOCS_TOKEN}"}
          - name: files
            command: npx
            args: [server-filesystem, data]
            cwd: data
        """,
    )
    monkeypatch.setenv("DOCS_TOKEN", "t0k")
    agent = Agent.from_harness(tmp_path, llm=_connection())
    docs, files = agent.mcp_servers

    assert isinstance(docs, MCPServerSse) and docs.name == "docs"
    assert docs.params["headers"] == {"Authorization": "Bearer t0k"}
    assert isinstance(files, MCPServerStdio)
    assert files.params["cwd"] == str((tmp_path / "data").resolve())


def test_an_mcp_server_needs_its_environment(tmp_path, _no_mcp_connection):
    _write(tmp_path, "harness.yaml", "mcp_servers:\n  - url: https://x/sse\n    headers: {A: '${NOT_SET_ANYWHERE}'}\n")
    h = Harness.load(tmp_path)
    with pytest.raises(HarnessError, match="NOT_SET_ANYWHERE"):
        h.build_agent(llm=_connection())


def test_an_mcp_server_has_one_transport(tmp_path):
    _write(tmp_path, "harness.yaml", "mcp_servers:\n  - url: https://x/sse\n    command: npx\n")
    with pytest.raises(HarnessError, match="either"):
        Harness.load(tmp_path)


# ---------------------------------------------------------------------- output schema


SCHEMA = '{"title": "Question", "type": "object", "properties": {"stem": {"type": "string"}}}'


def test_output_schema_becomes_the_response_model(tmp_path):
    (tmp_path / "output_schema.json").write_text(SCHEMA)
    agent = Agent.from_harness(tmp_path, llm=_connection())
    model = agent.response_model

    assert model.__name__ == "Question"
    assert model.model_json_schema()["properties"] == {"stem": {"type": "string"}}
    assert model.model_validate({"stem": "Which?"}).model_dump() == {"stem": "Which?"}

    strict = model.model_json_schema()
    strict["additionalProperties"] = False
    assert "additionalProperties" not in model.model_json_schema()  # a fresh copy each time


def test_a_response_model_in_code_wins(tmp_path):
    from pydantic import BaseModel

    class Answer(BaseModel):
        text: str

    (tmp_path / "output_schema.json").write_text(SCHEMA)
    assert Agent.from_harness(tmp_path, llm=_connection(), response_model=Answer).response_model is Answer


def test_output_schema_is_an_object_schema(tmp_path):
    (tmp_path / "output_schema.json").write_text('{"type": "array"}')
    with pytest.raises(HarnessError, match="object"):
        Harness.load(tmp_path)


# ---------------------------------------------------------------------- single-file agent


def test_from_file(tmp_path):
    _write(
        tmp_path,
        "reviewer.md",
        """
        ---
        description: Reviews questions.
        model: openai/gpt-6-luna
        model_settings: {temperature: 0}
        tools: [PubMedTool]
        ---
        You review {{ topic }} questions. <!-- not for the model -->
        """,
    )
    agent = Agent.from_file(tmp_path / "reviewer.md", llm=_connection(), variables={"topic": "cardiology"})

    assert agent.name == "reviewer"
    assert agent.system_prompt == "You review cardiology questions."
    assert agent.llm.model == "openai/gpt-6-luna" and agent.llm.temperature == 0
    assert "PubMedTool" in {t.name for t in agent.tools}
    assert Harness.load(tmp_path / "reviewer.md").name == "reviewer"


def test_from_file_refuses_unknown_keys(tmp_path):
    _write(tmp_path, "a.md", "---\nmodle: x\n---\nBody.")
    with pytest.raises(HarnessError, match="modle"):
        Agent.from_file(tmp_path / "a.md")


# ---------------------------------------------------------------------- CLI and flows


def test_cli_check_and_schema(harness_dir, tmp_path):
    import json

    from click.testing import CliRunner

    from gwenflow.cli import main

    result = CliRunner().invoke(main, ["harness", "check", str(harness_dir)])
    assert result.exit_code == 0, result.output
    assert "word_count" in result.output and "reviewer" in result.output

    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / "harness.yaml").write_text("modle: x\n")
    assert CliRunner().invoke(main, ["harness", "check", str(tmp_path / "bad")]).exit_code == 1

    schema = json.loads(CliRunner().invoke(main, ["harness", "schema"]).output)
    assert {"model", "tools", "mcp_servers", "approval"} <= set(schema["properties"])


def test_flow_agent_node_runs_a_harness(harness_dir, monkeypatch):
    from types import SimpleNamespace

    from gwenflow.flows.handlers import handle_agent

    seen = {}

    def fake_run(self, task, *args, **kwargs):
        seen.update(name=self.name, model=self.llm.model, task=task)
        return SimpleNamespace(content="done")

    monkeypatch.setattr(Agent, "run", fake_run)
    out = handle_agent({"harness": str(harness_dir), "model": "other", "task": "Topic: {topic}"}, {"topic": "DKA"})

    assert out == "done"
    assert seen == {"name": "question_generation", "model": "other", "task": "Topic: DKA"}
    with pytest.raises(ValueError, match="either"):
        handle_agent({"harness": str(harness_dir), "instructions": "x"}, None)
