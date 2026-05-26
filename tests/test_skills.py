"""Unit tests for the skills system.

Covers:
- Skill.from_directory / from_yaml / from_dict
- SkillsDirectory discovery
- SkillsToolset producing the three management tools (regression for the
  FunctionTool import bug that broke skill loading)
- Each management tool actually working (list / load / read resource)
- Skill.get_tools dynamically loading tools from script files
"""

import os
import textwrap
from pathlib import Path

import pytest

from gwenflow.skills import Skill, SkillsDirectory, SkillsToolset


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_skill_dir(root: Path, name: str, description: str, body: str = "Some instructions.", with_resource: bool = False, with_script: bool = False) -> Path:
    d = root / name
    d.mkdir()
    (d / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {description}\n---\n{body}")
    if with_resource:
        (d / "resources").mkdir()
        (d / "resources" / "facts.txt").write_text("The capital is Paris.")
    if with_script:
        (d / "scripts").mkdir()
        (d / "scripts" / "tools.py").write_text(
            textwrap.dedent(
                """
                def add_numbers(x: int, y: int) -> int:
                    \"\"\"Add two numbers and return the sum.\"\"\"
                    return x + y
                """
            ).strip()
        )
    return d


@pytest.fixture
def skill_root(tmp_path: Path) -> Path:
    _make_skill_dir(tmp_path, "weather", "Get weather information")
    _make_skill_dir(tmp_path, "news", "Get news headlines", with_resource=True)
    # A directory without SKILL.md should be silently ignored
    (tmp_path / "not_a_skill").mkdir()
    (tmp_path / "not_a_skill" / "README.md").write_text("just a readme")
    return tmp_path


# ---------------------------------------------------------------------------
# Skill.from_directory
# ---------------------------------------------------------------------------


def test_from_directory_parses_frontmatter(tmp_path: Path):
    d = _make_skill_dir(tmp_path, "demo", "A demo skill", body="Step 1. Step 2.")
    skill = Skill.from_directory(str(d))
    assert skill.name == "demo"
    assert skill.description == "A demo skill"
    assert "Step 1" in skill.content


def test_from_directory_missing_skill_md_raises(tmp_path: Path):
    d = tmp_path / "empty"
    d.mkdir()
    with pytest.raises(FileNotFoundError):
        Skill.from_directory(str(d))


def test_from_directory_collects_resources(tmp_path: Path):
    d = _make_skill_dir(tmp_path, "doc", "doc skill", with_resource=True)
    skill = Skill.from_directory(str(d))
    assert len(skill.resources) == 1
    assert skill.resources[0].endswith("facts.txt")


def test_from_directory_collects_scripts(tmp_path: Path):
    d = _make_skill_dir(tmp_path, "tooled", "with scripts", with_script=True)
    skill = Skill.from_directory(str(d))
    assert len(skill.scripts) == 1
    assert skill.scripts[0].endswith("tools.py")


def test_from_directory_ignores_private_scripts(tmp_path: Path):
    d = _make_skill_dir(tmp_path, "private", "test", with_script=True)
    (d / "scripts" / "_helper.py").write_text("def _x(): pass")
    skill = Skill.from_directory(str(d))
    assert all(not os.path.basename(s).startswith("_") for s in skill.scripts)


# ---------------------------------------------------------------------------
# Skill.from_dict / from_yaml
# ---------------------------------------------------------------------------


def test_from_dict_basic():
    skill = Skill.from_dict({"name": "x", "description": "y"})
    assert skill.name == "x"
    assert skill.description == "y"


def test_from_yaml(tmp_path: Path):
    p = tmp_path / "skill.yaml"
    p.write_text("name: from_yaml\ndescription: loaded via yaml\n")
    skill = Skill.from_yaml(str(p))
    assert skill.name == "from_yaml"


# ---------------------------------------------------------------------------
# Skill.to_prompt / to_dict
# ---------------------------------------------------------------------------


def test_to_prompt_includes_name_and_content():
    skill = Skill(name="demo", description="d", content="hello")
    prompt = skill.to_prompt()
    assert "<name>demo</name>" in prompt
    assert "hello" in prompt


def test_to_dict_omits_none_fields():
    skill = Skill(name="demo", description="d")
    d = skill.to_dict()
    assert "uri" not in d
    assert "license" not in d


# ---------------------------------------------------------------------------
# Skill.get_tools (script loading)
# ---------------------------------------------------------------------------


def test_get_tools_auto_wraps_documented_functions(tmp_path: Path):
    """Functions with docstrings + type annotations should be auto-wrapped as Tool."""
    d = _make_skill_dir(tmp_path, "math", "math skill", with_script=True)
    skill = Skill.from_directory(str(d))
    tools = skill.get_tools()
    assert len(tools) == 1
    assert tools[0].name == "add_numbers"
    # The wrapped tool actually runs (Tool.run stringifies, function gives int)
    assert tools[0].function(x=2, y=3) == 5


def test_get_tools_prefers_explicit_tools_list(tmp_path: Path):
    """If a script exports a module-level `tools` list, use it directly."""
    d = tmp_path / "explicit"
    d.mkdir()
    (d / "SKILL.md").write_text("---\nname: explicit\ndescription: x\n---\nbody")
    (d / "scripts").mkdir()
    (d / "scripts" / "main.py").write_text(
        textwrap.dedent(
            """
            from gwenflow.tools import Tool

            def multiply(x: int, y: int) -> int:
                \"\"\"Multiply two numbers.\"\"\"
                return x * y

            tools = [Tool(multiply)]
            """
        ).strip()
    )
    skill = Skill.from_directory(str(d))
    tools = skill.get_tools()
    assert len(tools) == 1
    assert tools[0].function(x=3, y=4) == 12


def test_get_tools_no_scripts_returns_empty():
    skill = Skill(name="x", description="y")
    assert skill.get_tools() == []


# ---------------------------------------------------------------------------
# SkillsDirectory
# ---------------------------------------------------------------------------


def test_directory_discovers_skills(skill_root: Path):
    sd = SkillsDirectory(str(skill_root))
    names = sorted(s.name for s in sd.skills)
    assert names == ["news", "weather"]


def test_directory_ignores_subdirs_without_skill_md(skill_root: Path):
    sd = SkillsDirectory(str(skill_root))
    # not_a_skill must be excluded
    assert "not_a_skill" not in [s.name for s in sd.skills]


def test_directory_missing_path_raises():
    with pytest.raises(NotADirectoryError):
        SkillsDirectory("/no/such/path/anywhere")


def test_directory_iter_and_len(skill_root: Path):
    sd = SkillsDirectory(str(skill_root))
    assert len(sd) == 2
    names = [s.name for s in sd]  # __iter__
    assert sorted(names) == ["news", "weather"]


# ---------------------------------------------------------------------------
# SkillsToolset — the three management tools (regression for the
# FunctionTool import bug that broke skill loading)
# ---------------------------------------------------------------------------


def test_toolset_get_tools_does_not_raise_import_error(skill_root: Path):
    """Regression: SkillsToolset.get_tools() used to fail with
    `ImportError: cannot import name 'FunctionTool' from 'gwenflow.tools'`."""
    sd = SkillsDirectory(str(skill_root))
    toolset = SkillsToolset(sd.skills)
    # This call is the one that used to crash
    tools = toolset.get_tools()
    assert len(tools) == 3


def test_toolset_returns_three_management_tools(skill_root: Path):
    sd = SkillsDirectory(str(skill_root))
    toolset = SkillsToolset(sd.skills)
    names = [t.name for t in toolset.get_tools()]
    assert names == ["list_skills", "load_skill", "read_skill_resource"]


def test_toolset_list_skills_returns_summary(skill_root: Path):
    sd = SkillsDirectory(str(skill_root))
    tools = SkillsToolset(sd.skills).get_tools()
    out = tools[0].run()
    assert "weather" in out and "news" in out


def test_toolset_load_skill_returns_full_instructions(skill_root: Path):
    sd = SkillsDirectory(str(skill_root))
    tools = SkillsToolset(sd.skills).get_tools()
    out = tools[1].run(skill_name="weather")
    assert "<name>weather</name>" in out


def test_toolset_load_skill_unknown_returns_helpful_message(skill_root: Path):
    sd = SkillsDirectory(str(skill_root))
    tools = SkillsToolset(sd.skills).get_tools()
    out = tools[1].run(skill_name="nope")
    assert "not found" in out.lower()
    assert "news" in out and "weather" in out  # lists available


def test_toolset_read_skill_resource(skill_root: Path):
    sd = SkillsDirectory(str(skill_root))
    tools = SkillsToolset(sd.skills).get_tools()
    out = tools[2].run(skill_name="news", resource_name="facts.txt")
    assert out == "The capital is Paris."


def test_toolset_read_resource_unknown_skill():
    toolset = SkillsToolset([])
    tools = toolset.get_tools()
    out = tools[2].run(skill_name="x", resource_name="y")
    assert "not found" in out.lower()


def test_toolset_get_instructions(skill_root: Path):
    sd = SkillsDirectory(str(skill_root))
    instr = SkillsToolset(sd.skills).get_instructions()
    assert "weather" in instr
    assert "news" in instr
    assert "<available_skills>" in instr


def test_toolset_get_instructions_empty():
    assert SkillsToolset([]).get_instructions() == ""


# ---------------------------------------------------------------------------
# Agent integration (lightweight, no LLM call)
# ---------------------------------------------------------------------------


def test_agent_with_skills_registers_management_tools(skill_root: Path):
    """Constructing an Agent with skills should register the three management
    tools without crashing — this is the path that broke in the user's
    internal_tests/test_skills.py."""
    from gwenflow import Agent, ChatOpenAI

    sd = SkillsDirectory(str(skill_root))
    agent = Agent(
        name="test",
        llm=ChatOpenAI(api_key="test"),
        skills=sd.skills,
    )
    tool_names = [t.name for t in agent.tools]
    assert "list_skills" in tool_names
    assert "load_skill" in tool_names
    assert "read_skill_resource" in tool_names
