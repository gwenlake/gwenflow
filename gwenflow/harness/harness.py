"""A harness: an agent described by a directory of files with fixed names.

```text
my_harness/
├── harness.yaml            settings (optional): model, model_settings, tools, mcp_servers, ...
├── system_prompt.md        the system prompt is these four files, in this order;
├── guardrails.md           a missing file is skipped, <!-- comments --> never reach the model
├── memory.md
├── tool_guidance.md
├── output_schema.json      the JSON Schema of the answer (optional)
├── skills/<name>/SKILL.md  skills (see SkillsDirectory)
├── subagents/<name>/AGENT.md   team members: frontmatter = settings, body = system prompt
└── tools/*.py              tools: a module-level `tools` list, else every public annotated
                            and documented function
```

Any other file is left to the application. A single-file agent is a markdown file whose
frontmatter holds the keys of ``harness.yaml`` and whose body is the system prompt.
``{{ name }}`` in the prose is replaced by the variables given when building the agent
(``date`` and ``datetime`` are always defined).

    agent = Agent.from_harness("my_harness", llm=ChatGwenlake(model="google/gemma-4-26b-a4b-it"))
    agent = Agent.from_file("reviewer.md")

    harness = Harness.load("my_harness")                  # read once,
    agent = harness.build_agent(variables={"level": 3})   # build one agent per run
"""

import copy
import dataclasses
import hashlib
import importlib.util
import inspect
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from gwenflow.agents.agent import Agent
from gwenflow.exceptions import UserError
from gwenflow.harness.config import SKILLS_INLINE, HarnessConfig, McpServerConfig, SubagentConfig, ToolEntry
from gwenflow.llms.base import ChatBase
from gwenflow.llms.gwenlake import ChatGwenlake
from gwenflow.skills import Skill, SkillsDirectory, SkillsToolset
from gwenflow.tools import BaseTool, Tool
from gwenflow.utils.markdown import parse_frontmatter

CONFIG_FILE = "harness.yaml"
PROSE_FILES = ("system_prompt.md", "guardrails.md", "memory.md", "tool_guidance.md")
"""Composed into the system prompt, in this order."""
OUTPUT_SCHEMA_FILE = "output_schema.json"
SKILLS_DIR = "skills"
SUBAGENTS_DIR = "subagents"
SUBAGENT_FILE = "AGENT.md"
TOOLS_DIR = "tools"

LIST_FIELDS = ("tools", "skills", "team", "mcp_servers")
"""`Agent` fields where what the code passes is added to what the harness holds."""

NOT_BUILTIN = {"BaseTool", "Tool"}
"""Names exported by `gwenflow.tools` that are not ready-made tools (nor are the `MCP*` ones)."""

_VARIABLE = re.compile(r"\{\{\s*([A-Za-z_]\w*)\s*\}\}")
_ENV = re.compile(r"\$\{([A-Za-z_]\w*)\}")
_COMMENT = re.compile(r"<!--.*?(-->|\Z)", re.DOTALL)


class HarnessError(UserError):
    """A harness that cannot be loaded or built."""


@dataclass
class Subagent:
    """``subagents/<name>/AGENT.md``, read."""

    name: str
    config: SubagentConfig
    system_prompt: str
    tools: List[BaseTool] = field(default_factory=list)


@dataclass
class Harness:
    """A harness, read once. ``build_agent`` makes a new `Agent` from it at each call."""

    path: Path
    """The harness directory, or the markdown file of a single-file agent."""
    config: HarnessConfig = field(default_factory=HarnessConfig)
    prose: Dict[str, str] = field(default_factory=dict)
    """Text of each prose file present, by file name, comments removed."""
    output_schema: Optional[Dict[str, Any]] = None
    skills: List[Skill] = field(default_factory=list)
    subagents: List[Subagent] = field(default_factory=list)
    tools: List[BaseTool] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.config.name or (self.path.stem if self.path.is_file() else self.path.name)

    @property
    def directory(self) -> Path:
        return self.path.parent if self.path.is_file() else self.path

    # ------------------------------------------------------------------ load

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Harness":
        """Read a harness directory (or its ``harness.yaml``), or a single-file agent (``.md``)."""
        path = Path(path).resolve()
        if path.is_file() and path.suffix == ".md":
            return cls.from_file(path)
        if path.is_file() and path.name == CONFIG_FILE:
            path = path.parent
        if not path.is_dir():
            raise HarnessError(f"'{path}' is not a harness directory")

        harness = cls(path=path, config=_read_config(path / CONFIG_FILE))
        for name in PROSE_FILES:
            if (path / name).is_file():
                harness.prose[name] = _read_text(path / name)
        harness.output_schema = _read_output_schema(path / OUTPUT_SCHEMA_FILE)
        if (path / SKILLS_DIR).is_dir():
            harness.skills = list(SkillsDirectory(str(path / SKILLS_DIR)).skills)
        harness.tools = _unique(_read_tools(path / TOOLS_DIR) + _builtin_tools(harness.config.tools, CONFIG_FILE))
        _require_approval(harness.tools, harness.config.approval, CONFIG_FILE)
        harness.subagents = _read_subagents(path / SUBAGENTS_DIR, harness)
        return harness

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "Harness":
        """A single-file agent: frontmatter = the keys of ``harness.yaml``, body = system prompt."""
        path = Path(path).resolve()
        if not path.is_file():
            raise HarnessError(f"'{path}' is not a file")
        frontmatter, body = parse_frontmatter(path.read_text(encoding="utf-8"))
        config = _validate(HarnessConfig, frontmatter, path.name)
        harness = cls(path=path, config=config)
        body = _COMMENT.sub("", body).strip()
        if body:
            harness.prose[PROSE_FILES[0]] = body
        harness.tools = _unique(_builtin_tools(config.tools, path.name))
        _require_approval(harness.tools, config.approval, path.name)
        return harness

    # ------------------------------------------------------------------ build

    def system_prompt(self, variables: Optional[Dict[str, Any]] = None) -> str:
        """The prose files in order, variables replaced, then the skills."""
        parts = [_render(text, variables, f"{self.name}/{name}") for name, text in self._ordered_prose()]
        parts.append(_skills_section(self.skills, self.config.skills))
        return "\n\n".join(p for p in parts if p.strip())

    def build_agent(
        self,
        *,
        variables: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        llm: Optional[ChatBase] = None,
        model_settings: Optional[Dict[str, Any]] = None,
        **agent_kwargs: Any,
    ) -> Agent:
        """A new `Agent` from this harness.

        The model is ``model``, else the one of the harness, else the one of ``llm``. It runs
        on a copy of ``llm`` (the connection: key, base URL, headers), else on `ChatGwenlake`.
        Then what the code passes wins over the files: ``model_settings`` are set after those
        of the harness, and other ``Agent`` arguments replace the harness value, except
        ``tools``, ``skills``, ``team`` and ``mcp_servers``, which are added to it.
        """
        base = self._llm(model or self.config.model, llm)
        _apply_settings(base, self.config.model_settings, self.name)
        _apply_settings(base, model_settings or {}, "model_settings")

        params: Dict[str, Any] = {
            "name": self.name,
            "description": self.config.description,
            "system_prompt": self.system_prompt(variables),
            "llm": base,
            "tools": list(self.tools),
            "skills": self._agent_skills(self.skills),
            "team": [self._build_subagent(s, base, variables) for s in self.subagents],
            "mcp_servers": [_mcp_server(c, self.directory) for c in self.config.mcp_servers],
        }
        if self.output_schema is not None:
            params["response_model"] = schema_model(self.output_schema)
        if self.config.max_turns is not None:
            params["max_turns"] = self.config.max_turns
        for key, value in agent_kwargs.items():
            params[key] = list(params.get(key) or []) + list(value or []) if key in LIST_FIELDS else value
        params["team"] = params["team"] or None
        return Agent(**params)

    def _ordered_prose(self) -> List[tuple]:
        return [(name, self.prose[name]) for name in PROSE_FILES if name in self.prose]

    def _agent_skills(self, skills: List[Skill]) -> List[Skill]:
        """Inline skills are in the system prompt already: the agent gets no skill tool for them."""
        return [] if self.config.skills == SKILLS_INLINE else list(skills)

    def _llm(self, model: Optional[str], base: Optional[ChatBase]) -> ChatBase:
        """A copy of ``base`` with ``model`` if given; without ``base``, a `ChatGwenlake` for ``model``."""
        if base is None:
            if model is None:
                raise HarnessError(
                    f"'{self.name}' has no model: set `model` in its settings, or pass `model=` or `llm=`"
                )
            return ChatGwenlake(model=model)
        llm = copy.copy(base)
        if model is not None:
            llm.model = model
        return llm

    def _build_subagent(self, sub: Subagent, parent: ChatBase, variables: Optional[Dict[str, Any]]) -> Agent:
        where = f"{SUBAGENTS_DIR}/{sub.name}/{SUBAGENT_FILE}"
        llm = self._llm(sub.config.model, parent)
        _apply_settings(llm, sub.config.model_settings, where)
        skills = [s for s in self.skills if s.name in sub.config.skills]
        prompt = "\n\n".join(
            p
            for p in (_render(sub.system_prompt, variables, where), _skills_section(skills, self.config.skills))
            if p.strip()
        )
        kwargs: Dict[str, Any] = {}
        if sub.config.max_turns is not None:
            kwargs["max_turns"] = sub.config.max_turns
        return Agent(
            name=sub.name,
            description=sub.config.description,
            system_prompt=prompt or None,
            llm=llm,
            tools=list(sub.tools),
            skills=self._agent_skills(skills),
            **kwargs,
        )

    def __repr__(self) -> str:
        return (
            f"Harness(name={self.name!r}, prose={list(self.prose)}, skills={[s.name for s in self.skills]}, "
            f"subagents={[s.name for s in self.subagents]}, tools={[t.name for t in self.tools]}, "
            f"mcp_servers={len(self.config.mcp_servers)}, output_schema={self.output_schema is not None})"
        )


# ---------------------------------------------------------------------- structured output


def schema_model(schema: Dict[str, Any]) -> Type[BaseModel]:
    """A `response_model` from a JSON Schema.

    The schema goes to the model as is, and the answer comes back parsed, its fields as
    attributes (``.model_dump()`` gives the dict). The answer is not validated against the schema.
    """
    frozen = copy.deepcopy(schema)
    name = re.sub(r"[^A-Za-z0-9_-]", "_", str(schema.get("title") or "Output"))

    class _SchemaModel(BaseModel):
        model_config = ConfigDict(extra="allow")

        @classmethod
        def model_json_schema(cls, *args: Any, **kwargs: Any) -> Dict[str, Any]:
            return copy.deepcopy(frozen)  # callers make it strict in place

    _SchemaModel.__name__ = _SchemaModel.__qualname__ = name
    return _SchemaModel


# ---------------------------------------------------------------------- reading


def _read_text(path: Path) -> str:
    return _COMMENT.sub("", path.read_text(encoding="utf-8")).strip()


def _validate(model: Type[BaseModel], raw: Any, where: str) -> Any:
    if not isinstance(raw, dict):
        raise HarnessError(f"{where}: expected a mapping")
    try:
        return model.model_validate(raw)
    except ValidationError as e:
        errors = "; ".join(f"{'.'.join(str(p) for p in x['loc']) or 'file'}: {x['msg']}" for x in e.errors())
        raise HarnessError(f"{where}: {errors}") from e


def _read_config(path: Path) -> HarnessConfig:
    if not path.is_file():
        return HarnessConfig()
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        raise HarnessError(f"{path}: invalid YAML: {e}") from e
    return _validate(HarnessConfig, raw, str(path))


def _read_output_schema(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        schema = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise HarnessError(f"{OUTPUT_SCHEMA_FILE}: invalid JSON: {e}") from e
    if not isinstance(schema, dict) or schema.get("type") != "object":
        raise HarnessError(f'{OUTPUT_SCHEMA_FILE}: expected a JSON Schema with "type": "object"')
    return schema


def _read_subagents(directory: Path, harness: Harness) -> List[Subagent]:
    if not directory.is_dir():
        return []
    skill_names = {s.name for s in harness.skills}
    subagents: List[Subagent] = []
    for agent_md in sorted(directory.glob(f"*/{SUBAGENT_FILE}")):
        where = f"{SUBAGENTS_DIR}/{agent_md.parent.name}/{SUBAGENT_FILE}"
        frontmatter, body = parse_frontmatter(agent_md.read_text(encoding="utf-8"))
        config = _validate(SubagentConfig, frontmatter, where)
        unknown = sorted(set(config.skills) - skill_names)
        if unknown:
            raise HarnessError(f"{where}: unknown skills {unknown}; the harness has {sorted(skill_names)}")
        name = config.name or agent_md.parent.name
        if any(s.name == name for s in subagents):
            raise HarnessError(f"{where}: another subagent is already named '{name}'")
        subagents.append(
            Subagent(
                name=name,
                config=config,
                system_prompt=_COMMENT.sub("", body).strip(),
                tools=_subagent_tools(config.tools, harness.tools, where),
            )
        )
    return subagents


def _subagent_tools(names: List[str], harness_tools: List[BaseTool], where: str) -> List[BaseTool]:
    """A harness tool by name, else a built-in tool."""
    by_name = {t.name: t for t in harness_tools}
    return [by_name[n] if n in by_name else _builtin_tool(n, {}, where) for n in names]


def _read_tools(directory: Path) -> List[BaseTool]:
    if not directory.is_dir():
        return []
    scope = "gwenflow_harness_" + hashlib.sha256(str(directory).encode()).hexdigest()[:12]
    tools: List[BaseTool] = []
    for script in sorted(directory.glob("*.py")):
        if script.name.startswith("_"):
            continue
        module = _import(f"{scope}.{script.stem}", script)
        explicit = getattr(module, "tools", None)
        if isinstance(explicit, list):
            tools.extend(explicit)
            continue
        for fn_name, fn in inspect.getmembers(module, inspect.isfunction):
            if fn_name.startswith("_") or fn.__module__ != module.__name__ or not (fn.__doc__ and fn.__annotations__):
                continue
            try:
                tools.append(Tool(fn))
            except Exception as e:
                raise HarnessError(f"{TOOLS_DIR}/{script.name}: cannot make a tool of {fn_name}: {e}") from e
    return tools


def _import(module_name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise HarnessError(f"{TOOLS_DIR}/{path.name}: cannot be imported")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module  # so its functions can be pickled and introspected
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        sys.modules.pop(module_name, None)
        raise HarnessError(f"{TOOLS_DIR}/{path.name}: {type(e).__name__}: {e}") from e
    return module


def _builtin_tools(entries: List[ToolEntry], where: str) -> List[BaseTool]:
    tools: List[BaseTool] = []
    for entry in entries:
        if isinstance(entry, str):
            tools.append(_builtin_tool(entry, {}, where))
        elif len(entry) == 1:
            ((name, kwargs),) = entry.items()
            tools.append(_builtin_tool(name, kwargs or {}, where))
        else:
            raise HarnessError(f"{where}: a tool entry is a name or {{Name: {{...}}}}, got {sorted(entry)}")
    return tools


def _builtin_tool(name: str, kwargs: Dict[str, Any], where: str) -> BaseTool:
    import gwenflow.tools as builtins

    if name in NOT_BUILTIN or name.startswith("MCP") or name not in builtins.__all__:
        available = sorted(n for n in builtins.__all__ if n not in NOT_BUILTIN and not n.startswith("MCP"))
        raise HarnessError(f"{where}: unknown tool '{name}'; built-in tools are {available}")
    try:
        return getattr(builtins, name)(**kwargs)
    except Exception as e:
        raise HarnessError(f"{where}: cannot create {name}: {type(e).__name__}: {e}") from e


def _unique(tools: List[BaseTool]) -> List[BaseTool]:
    seen: set = set()
    for tool in tools:
        if tool.name in seen:
            raise HarnessError(f"two tools are named '{tool.name}'")
        seen.add(tool.name)
    return tools


def _require_approval(tools: List[BaseTool], names: List[str], where: str) -> None:
    unknown = sorted(set(names) - {t.name for t in tools})
    if unknown:
        raise HarnessError(f"{where}: `approval` names unknown tools {unknown}")
    for tool in tools:
        if tool.name in names:
            tool.requires_approval = True


# ---------------------------------------------------------------------- building


def _render(text: str, variables: Optional[Dict[str, Any]], where: str) -> str:
    now = datetime.now(timezone.utc)
    values = {"date": now.strftime("%Y-%m-%d"), "datetime": now.isoformat(timespec="seconds"), **(variables or {})}
    missing = sorted({m for m in _VARIABLE.findall(text) if m not in values})
    if missing:
        raise HarnessError(f"{where} uses {missing}: pass them in `variables`")
    return _VARIABLE.sub(lambda m: str(values[m.group(1)]), text)


def _skills_section(skills: List[Skill], mode: str) -> str:
    if not skills:
        return ""
    if mode == SKILLS_INLINE:
        blocks = "\n\n".join(
            f'<skill name="{s.name}">\n{s.description}\n\n{s.content.strip()}\n</skill>' for s in skills
        )
        return "## Skills\n\nApply the skill whose situation matches.\n\n" + blocks
    return "## Skills:\n\n" + SkillsToolset(skills).get_instructions()


def _apply_settings(llm: ChatBase, settings: Dict[str, Any], where: str) -> None:
    allowed = {f.name for f in dataclasses.fields(llm)} if dataclasses.is_dataclass(llm) else set(vars(llm))
    allowed -= {"model", "tools", "tool_choice", "client", "async_client", "response_format"}
    unknown = sorted(set(settings) - allowed)
    if unknown:
        raise HarnessError(f"{where}: {type(llm).__name__} has no model setting {unknown}")
    for key, value in settings.items():
        setattr(llm, key, value)


def _env(value: Any, where: str) -> Any:
    """``${NAME}`` replaced by the environment variable ``NAME``, in a string, list or dict."""
    if isinstance(value, str):

        def lookup(m: re.Match) -> str:
            if m.group(1) not in os.environ:
                raise HarnessError(f"{where}: environment variable {m.group(1)} is not set")
            return os.environ[m.group(1)]

        return _ENV.sub(lookup, value)
    if isinstance(value, list):
        return [_env(v, where) for v in value]
    if isinstance(value, dict):
        return {k: _env(v, where) for k, v in value.items()}
    return value


def _mcp_server(config: McpServerConfig, directory: Path) -> Any:
    from gwenflow.tools.mcp import MCPServerSse, MCPServerStdio

    where = f"mcp_servers.{config.name or config.url or config.command}"
    if config.url is not None:
        params: Dict[str, Any] = {"url": _env(config.url, where)}
        if config.headers:
            params["headers"] = _env(config.headers, where)
        if config.timeout is not None:
            params["timeout"] = config.timeout
        if config.sse_read_timeout is not None:
            params["sse_read_timeout"] = config.sse_read_timeout
        return MCPServerSse(params, name=config.name)
    params = {"command": config.command, "args": _env(config.args, where)}
    if config.env:
        params["env"] = _env(config.env, where)
    if config.cwd is not None:
        params["cwd"] = str((directory / config.cwd).resolve())
    return MCPServerStdio(params, name=config.name)
