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
    assert list(h.prompt_parts()) == ["instructions", "guardrails", "tool_guidance"]
    assert h.config.guardrails == "Never give the answer in the stem."
    assert "note for editors" not in h.config.instructions
    assert [s.name for s in h.skills] == ["distractor-design"]
    assert [t.name for t in h.tools] == ["word_count"]
    assert [s.name for s in h.subagents] == ["reviewer"]


def test_load_accepts_the_config_file_path(harness_dir):
    assert Harness.load(harness_dir / "harness.yaml").path == harness_dir


def test_an_empty_directory_is_a_harness(tmp_path):
    h = Harness.load(tmp_path)
    assert h.name == tmp_path.name
    assert h.instructions() == ""


@pytest.mark.parametrize(
    "config, message",
    [
        ("modle: x\n", "modle"),
        ("skills: sometimes\n", "skills"),
        ("[a, b]\n", "expected a mapping"),
        ("name: [\n", "invalid file"),
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
    with pytest.raises(HarnessError, match="unknown capability or tool .search."):
        Harness.load(tmp_path)


# ---------------------------------------------------------------------- system prompt


def test_system_prompt_follows_the_fixed_order(harness_dir):
    prompt = Agent.from_harness(harness_dir).get_system_prompt(task="hi")

    assert prompt.index("You write questions") < prompt.index("Never give") < prompt.index("Use word_count")
    assert "{{" not in prompt
    assert "<name>distractor-design</name>" in prompt  # skills index, loaded on demand


def test_variables_fill_the_prose(tmp_path):
    _write(tmp_path, "system_prompt.md", "Level {{ level }}.")
    h = Harness.load(tmp_path)

    assert h.instructions({"level": 3}) == "Level 3."
    with pytest.raises(HarnessError, match="level"):
        h.instructions()


def test_inline_skills_go_into_the_prompt_without_tools(harness_dir):
    config = harness_dir / "harness.yaml"
    config.write_text(config.read_text() + "capabilities: [{Skills: {mode: inline}}]\n")
    agent = Agent.from_harness(harness_dir, llm=_connection())

    assert "Each distractor is a real misconception." in agent.instructions
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
    assert reviewer.instructions == "You review medical exam questions."
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


# ---------------------------------------------------------------------- capabilities (Pydantic AI syntax)


def test_a_pydantic_ai_spec_loads_as_is(tmp_path, _no_mcp_connection):
    _write(
        tmp_path,
        "agent.yaml",
        """
        model: google/gemma-4-26b-a4b-it
        instructions: "You are helping {{user_name}}."
        model_settings:
          max_tokens: 8192
        capabilities:
          - Thinking:
              effort: high
          - MCP: https://mcp.example.com/sse
        output_schema:
          type: object
          properties:
            answer: {type: string}
        """,
    )
    agent = Agent.from_file(tmp_path / "agent.yaml", llm=_connection(), variables={"user_name": "Ada"})

    assert agent.name == "agent"
    assert agent.instructions == "You are helping Ada."
    assert agent.llm.max_tokens == 8192 and agent.llm.reasoning_effort == "high"
    assert len(agent.mcp_servers) == 1
    assert agent.response_model.model_json_schema()["properties"] == {"answer": {"type": "string"}}


def test_tools_as_capabilities(tmp_path):
    _write(
        tmp_path,
        "harness.yaml",
        "capabilities:\n  - ClinicalTrialsTool\n  - PubMedTool: {max_results: 5, requires_approval: true}\n",
    )
    h = Harness.load(tmp_path)

    assert [t.name for t in h.tools] == ["ClinicalTrialsTool", "PubMedTool"]
    assert h.tools[1].max_results == 5 and h.tools[1].requires_approval


@pytest.mark.parametrize("name", ["NoSuchTool", "Tool", "MCPServerSse"])
def test_unknown_capabilities_are_refused(tmp_path, name):
    _write(tmp_path, "harness.yaml", f"capabilities: [{name}]\n")
    with pytest.raises(HarnessError, match="unknown capability"):
        Harness.load(tmp_path)


def test_thinking_then_model_settings(tmp_path):
    _write(tmp_path, "harness.yaml", "capabilities: [{Thinking: low}]\nmodel_settings: {reasoning_effort: none}\n")
    assert Agent.from_harness(tmp_path, llm=_connection()).llm.reasoning_effort == "none"


def test_skills_capability(tmp_path):
    for folder, name in (("skills", "local-one"), ("shared/skills", "shared-one"), ("shared/skills", "other")):
        _write(tmp_path, f"{folder}/{name}/SKILL.md", f"---\nname: {name}\ndescription: {name}.\n---\nBody.")
    _write(
        tmp_path,
        "harness.yaml",
        "capabilities:\n  - Skills: {directories: shared/skills, exclude: [other], mode: inline}\n",
    )
    h = Harness.load(tmp_path)

    assert [s.name for s in h.skills] == ["local-one", "shared-one"]
    assert h.skills_inline


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


def test_mcp_capabilities(tmp_path, monkeypatch, _no_mcp_connection):
    from gwenflow.tools.mcp import MCPServerSse, MCPServerStdio

    _write(
        tmp_path,
        "harness.yaml",
        """
        capabilities:
          - MCP: {id: docs, url: https://docs.example.com/sse, headers: {Authorization: "Bearer ${DOCS_TOKEN}"}}
          - MCP: {id: files, command: npx, args: [server-filesystem, data], cwd: data}
        """,
    )
    monkeypatch.setenv("DOCS_TOKEN", "t0k")
    docs, files = Agent.from_harness(tmp_path, llm=_connection()).mcp_servers

    assert isinstance(docs, MCPServerSse) and docs.params["headers"] == {"Authorization": "Bearer t0k"}
    assert isinstance(files, MCPServerStdio) and files.params["cwd"] == str((tmp_path / "data").resolve())

    _write(
        tmp_path,
        "harness.yaml",
        "capabilities:\n  - MCP: {url: 'https://x/sse', headers: {A: '${NOT_SET_ANYWHERE}'}}\n",
    )
    with pytest.raises(HarnessError, match="NOT_SET_ANYWHERE"):
        Agent.from_harness(tmp_path, llm=_connection())


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
        capabilities: [PubMedTool]
        ---
        You review {{ topic }} questions. <!-- not for the model -->
        """,
    )
    agent = Agent.from_file(tmp_path / "reviewer.md", llm=_connection(), variables={"topic": "cardiology"})

    assert agent.name == "reviewer"
    assert agent.instructions == "You review cardiology questions."
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
    assert {"model", "instructions", "capabilities", "output_schema"} <= set(schema["properties"])


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


# ---------------------------------------------------------------------- long-term memory


def test_knowledge_and_long_term_memory_together(harness_dir, tmp_path):
    import asyncio

    from gwenflow.memory import MemoryToolset

    _write(harness_dir, "knowledge.md", "Potassium above 5.5 mmol/L is hyperkalaemia.")
    config = harness_dir / "harness.yaml"
    store = tmp_path / "memories"
    config.write_text(
        config.read_text()
        + f"""long_term_memory:
  path: {store}/{{{{ user_id }}}}
  always: [/memories/user/MEMORY.md]
  tools: [read_memory, memory_edit]
"""
    )
    agent = Agent.from_harness(harness_dir, llm=_connection(), variables={"user_id": "u42"})

    assert isinstance(agent.memory, MemoryToolset)
    assert agent.memory.store.base_dir == (store / "u42").resolve()
    assert {"read_memory", "memory_edit"} <= {t.name for t in agent.tools}

    asyncio.run(agent.memory.aload())
    prompt = agent.get_system_prompt(task="hi")
    assert "Potassium above 5.5" in prompt  # knowledge: fixed facts, in the prompt parts
    assert prompt.index("You write questions") < prompt.index("## Memory:")  # long-term memory, each run


def test_long_term_memory_needs_a_target(tmp_path):
    _write(tmp_path, "harness.yaml", "long_term_memory: {store: local}\n")
    with pytest.raises(HarnessError, match="path"):
        Harness.load(tmp_path)


def test_a_memory_given_in_code_wins(harness_dir, tmp_path):
    from gwenflow.memory import MemoryToolset
    from gwenflow.stores.files import InMemoryMemoryStore

    config = harness_dir / "harness.yaml"
    config.write_text(config.read_text() + f"long_term_memory: {{path: {tmp_path}/never}}\n")
    memory = MemoryToolset(InMemoryMemoryStore())

    assert Agent.from_harness(harness_dir, llm=_connection(), memory=memory).memory is memory
    assert not (tmp_path / "never").exists()


def test_the_run_context_reaches_the_prompt(harness_dir):
    agent = Agent.from_harness(harness_dir, llm=_connection())
    assert "level 3 students" in agent.get_system_prompt(task="hi", context="level 3 students")


# ---------------------------------------------------------------------- prompt parts: in harness.yaml or in their file


def test_prompt_parts_in_the_definition_or_in_files(tmp_path):
    _write(
        tmp_path,
        "harness.yaml",
        """
        instructions:
          - You write exam questions.
          - One correct answer only.
        knowledge: Potassium above 5.5 mmol/L is hyperkalaemia.
        """,
    )
    _write(tmp_path, "guardrails.md", "Never give the answer in the stem.")
    h = Harness.load(tmp_path)

    assert h.config.guardrails == "Never give the answer in the stem."  # from its file
    assert list(h.prompt_parts()) == ["instructions", "guardrails", "knowledge"]
    assert h.instructions() == (
        "You write exam questions.\n\nOne correct answer only.\n\n"
        "Never give the answer in the stem.\n\n"
        "Potassium above 5.5 mmol/L is hyperkalaemia."
    )


@pytest.mark.parametrize("part, filename", [("instructions", "system_prompt.md"), ("knowledge", "knowledge.md")])
def test_a_prompt_part_is_written_once(tmp_path, part, filename):
    _write(tmp_path, "harness.yaml", f"{part}: In the definition.\n")
    _write(tmp_path, filename, "In the file.")
    with pytest.raises(HarnessError, match=f"`{part}` is written in both"):
        Harness.load(tmp_path)


def test_a_subagent_prompt_in_its_frontmatter(tmp_path):
    _write(tmp_path, "subagents/x/AGENT.md", "---\ndescription: d\ninstructions: Check facts.\n---\n")
    (sub,) = Harness.load(tmp_path).subagents
    assert sub.config.instructions == "Check facts."

    _write(tmp_path, "subagents/x/AGENT.md", "---\ndescription: d\ninstructions: Check facts.\n---\nAnd a body.")
    with pytest.raises(HarnessError, match="written in both"):
        Harness.load(tmp_path)


# ---------------------------------------------------------------------- application skills


SCRIPT_TOOL = '''
def stock_level(sku: str) -> int:
    """The stock of an article.

    Args:
        sku: The article reference.
    """
    return 0
'''


def _app_harness(root: Path, *, script_tools: bool = True, templating: str = "plain") -> Path:
    for name in ("quote", "order", "task"):
        _write(root, f"skills/{name}/SKILL.md", f"---\nname: {name}\ndescription: The {name}s.\n---\nBody of {name}.")
    _write(root, "skills/order/scripts/tool.py", SCRIPT_TOOL)
    _write(
        root,
        "harness.yaml",
        f"""
        templating: {templating}
        capabilities:
          - Skills: {{script_tools: {str(script_tools).lower()}}}
        """,
    )
    return root


def test_only_skills_keeps_the_named_ones(tmp_path):
    h = Harness.load(_app_harness(tmp_path))
    agent = h.build_agent(llm=_connection(), only_skills=["task", "quote"])

    assert [s.name for s in agent.skills] == ["quote", "task"]
    assert [s.name for s in h.build_agent(llm=_connection()).skills] == ["order", "quote", "task"]
    assert h.build_agent(llm=_connection(), only_skills=[]).skills == []


def test_only_skills_refuses_an_unknown_name(tmp_path):
    with pytest.raises(HarnessError, match="no skill named \\['invoice'\\]"):
        Harness.load(_app_harness(tmp_path)).build_agent(llm=_connection(), only_skills=["quote", "invoice"])


def test_only_skills_applies_to_the_subagents(tmp_path):
    _app_harness(tmp_path)
    _write(tmp_path, "subagents/sales/AGENT.md", "---\ndescription: Sales.\nskills: [quote, order]\n---\nSell.")
    agent = Harness.load(tmp_path).build_agent(llm=_connection(), only_skills=["quote", "task"])

    (sales,) = agent.team
    assert [s.name for s in sales.skills] == ["quote"]


def test_skill_script_tools_are_registered_by_default(tmp_path):
    agent = Harness.load(_app_harness(tmp_path)).build_agent(llm=_connection())

    assert "stock_level" in {t.name for t in agent.tools}


def test_skill_script_tools_can_be_left_to_the_application(tmp_path):
    _app_harness(tmp_path, script_tools=False)
    _write(tmp_path, "subagents/sales/AGENT.md", "---\ndescription: Sales.\nskills: [order]\n---\nSell.")
    h = Harness.load(tmp_path)
    agent = h.build_agent(llm=_connection())

    assert not h.skill_script_tools
    names = {t.name for t in agent.tools}
    assert "stock_level" not in names
    assert "load_skill" in names  # the skills themselves stay loadable
    assert "stock_level" not in {t.name for t in agent.team[0].tools}


def test_jinja_templating_sees_the_skills_and_the_variables(tmp_path):
    _app_harness(tmp_path, templating="jinja")
    _write(
        tmp_path,
        "system_prompt.md",
        """
        Hello {{ user }}.
        {% if 'quote' in skills %}
        * QUOTE: a priced offer.
        {% endif %}
        {% if 'order' in skills %}
        * ORDER: a confirmed purchase.
        {% endif %}
        <!-- {% if broken %} comments never reach Jinja -->
        """,
    )
    agent = Harness.load(tmp_path).build_agent(llm=_connection(), only_skills=["quote"], variables={"user": "Ada"})

    assert agent.instructions == "Hello Ada.\n* QUOTE: a priced offer."


def test_jinja_templating_refuses_an_undefined_variable(tmp_path):
    _app_harness(tmp_path, templating="jinja")
    _write(tmp_path, "system_prompt.md", "Hello {{ user }}.")

    with pytest.raises(HarnessError, match="'user' is undefined"):
        Harness.load(tmp_path).build_agent(llm=_connection())


def test_jinja_templating_renders_the_subagents(tmp_path):
    _app_harness(tmp_path, templating="jinja")
    _write(
        tmp_path,
        "subagents/sales/AGENT.md",
        "---\ndescription: Sales.\nskills: [quote, order]\n---\nSell{% for s in skills %} {{ s }}{% endfor %}.",
    )
    agent = Harness.load(tmp_path).build_agent(llm=_connection(), only_skills=["order"])

    assert agent.team[0].instructions == "Sell order."


def test_plain_templating_leaves_jinja_blocks_alone(tmp_path):
    _write(tmp_path, "system_prompt.md", "Keep {% raw %} as written.")

    assert Harness.load(tmp_path).build_agent(llm=_connection()).instructions == "Keep {% raw %} as written."
