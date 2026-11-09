"""A harness: an agent described by files, with the key names of the Pydantic AI agent spec.

```text
my_harness/
├── harness.yaml            the definition (optional), see `HarnessConfig`
├── system_prompt.md        the parts of the prompt, in this order; each one in harness.yaml
├── guardrails.md           (`instructions`, `guardrails`, `knowledge`, `tool_guidance`) or in
├── knowledge.md            its file, not both; <!-- comments --> never reach the model
├── tool_guidance.md
├── output_schema.json      the JSON Schema of the answer (or `output_schema` in harness.yaml)
├── skills/<name>/SKILL.md  skills
├── subagents/<name>/AGENT.md   team members: frontmatter = definition, body = instructions
└── tools/*.py              tools: a module-level `tools` list, else every public annotated
                            and documented function
```

An agent can also be one file: a Pydantic AI spec (``agent.yaml``, ``.json``), or a markdown
file whose frontmatter is the definition and whose body is its ``instructions``. ``{{ name }}``
in the prompt is replaced by ``variables`` (``date`` and ``datetime`` are always defined).

    agent = Agent.from_harness("my_harness", llm=ChatGwenlake(model="google/gemma-4-26b-a4b-it"))
    agent = Agent.from_file("agent.yaml")
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
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from gwenflow.agents.agent import Agent
from gwenflow.harness.config import (
    Capability,
    HarnessConfig,
    HarnessError,
    LongTermMemoryConfig,
    McpCapability,
    SkillsCapability,
    SubagentConfig,
)
from gwenflow.llms.base import ChatBase
from gwenflow.llms.gwenlake import ChatGwenlake
from gwenflow.skills import Skill, SkillsDirectory
from gwenflow.tools import BaseTool, Tool
from gwenflow.utils.markdown import parse_frontmatter

CONFIG_FILE = "harness.yaml"
PROMPT_PARTS = (
    ("instructions", "system_prompt.md"),
    ("guardrails", "guardrails.md"),
    ("knowledge", "knowledge.md"),
    ("tool_guidance", "tool_guidance.md"),
)
OUTPUT_SCHEMA_FILE = "output_schema.json"
CAPABILITY_TOOLS = {"WebSearch": "DuckDuckGoSearchTool", "WebFetch": "WebsiteReaderTool"}
"""Pydantic AI capabilities that a gwenflow tool provides."""
ADDED_FIELDS = ("tools", "skills", "team", "mcp_servers")
"""`Agent` fields where what the code passes is added to what the harness holds, not replaced."""

_VARIABLE = re.compile(r"\{\{\s*([A-Za-z_]\w*)\s*\}\}")
_ENV_VARIABLE = re.compile(r"\$\{([A-Za-z_]\w*)\}")
_COMMENT = re.compile(r"<!--.*?(-->|\Z)", re.DOTALL)


@dataclass
class Subagent:
    name: str
    config: SubagentConfig
    tools: List[BaseTool] = field(default_factory=list)


@dataclass
class Harness:
    """A harness, read once. ``build_agent`` makes a new `Agent` from it at each call."""

    path: Path
    """The harness directory, or the agent file."""
    config: HarnessConfig = field(default_factory=HarnessConfig)
    """The definition, prompt parts and output schema included, whatever file they came from."""
    tools: List[BaseTool] = field(default_factory=list)
    skills: List[Skill] = field(default_factory=list)
    skills_inline: bool = False
    """The skills are written in full in the prompt, instead of loaded on demand by a tool."""
    mcp_servers: List[McpCapability] = field(default_factory=list)
    reasoning_effort: Optional[str] = None
    """Set by the ``Thinking`` capability."""
    subagents: List[Subagent] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.config.name or (self.path.stem if self.path.is_file() else self.path.name)

    @property
    def directory(self) -> Path:
        return self.path.parent if self.path.is_file() else self.path

    # ------------------------------------------------------------------ load

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Harness":
        """Read a harness directory (or its ``harness.yaml``), or an agent file."""
        path = Path(path).resolve()
        if path.is_file():
            if path.name != CONFIG_FILE:
                return cls.from_file(path)
            path = path.parent
        if not path.is_dir():
            raise HarnessError(f"'{path}' is not a harness directory")

        config_file = path / CONFIG_FILE
        config = (
            _parse(HarnessConfig, _read_data(config_file), CONFIG_FILE) if config_file.is_file() else HarnessConfig()
        )
        for part, filename in PROMPT_PARTS:
            if (path / filename).is_file():
                _set_from_file(config, part, _read_markdown(path / filename), filename)
        if (path / OUTPUT_SCHEMA_FILE).is_file():
            _set_from_file(config, "output_schema", _read_data(path / OUTPUT_SCHEMA_FILE), OUTPUT_SCHEMA_FILE)

        harness = cls._from_config(path, config, CONFIG_FILE)
        harness.subagents = _read_subagents(path / "subagents", harness.tools)
        return harness

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "Harness":
        """An agent file: a Pydantic AI spec (``.yaml``, ``.json``) or a markdown file."""
        path = Path(path).resolve()
        if not path.is_file():
            raise HarnessError(f"'{path}' is not a file")
        if path.suffix == ".md":
            frontmatter, body = parse_frontmatter(path.read_text(encoding="utf-8"))
            config = _parse(HarnessConfig, frontmatter, path.name)
            _set_from_file(config, "instructions", _strip_comments(body), f"the body of {path.name}")
        else:
            config = _parse(HarnessConfig, _read_data(path), path.name)
        return cls._from_config(path, config, path.name)

    @classmethod
    def _from_config(cls, path: Path, config: HarnessConfig, where: str) -> "Harness":
        """The harness of a definition: its capabilities read, its tools and skills loaded."""
        harness = cls(path=path, config=config)
        tools = _read_tools(path / "tools") if path.is_dir() else []
        skills_options = SkillsCapability()

        for entry in config.capabilities:
            name, value = _capability(entry, where)
            here = f"{where}: {name}"
            if name == "Skills":
                skills_options = _parse(SkillsCapability, _kwargs(value, "directories", here), here)
            elif name == "MCP":
                harness.mcp_servers.append(_parse(McpCapability, _kwargs(value, "url", here), here))
            elif name == "Thinking":
                harness.reasoning_effort = _kwargs(value, "effort", here).get("effort", "medium")
            else:
                tools.append(builtin_tool(CAPABILITY_TOOLS.get(name, name), _kwargs(value, None, here), here))

        harness.tools = _unique(tools, where)
        skills_folder = path / "skills" if path.is_dir() else None
        harness.skills = _read_skills(skills_folder, harness.directory, skills_options)
        harness.skills_inline = skills_options.mode == "inline"
        return harness

    # ------------------------------------------------------------------ build

    def prompt_parts(self) -> Dict[str, str]:
        """The parts of the prompt that have a text, in order."""
        texts = {part: _prompt_text(getattr(self.config, part)) for part, _ in PROMPT_PARTS}
        return {part: text for part, text in texts.items() if text}

    def instructions(self, variables: Optional[Dict[str, Any]] = None) -> str:
        """The parts of the prompt, variables replaced, then the inline skills.

        The agent then adds what it renders at each run: the skills index, the long-term memory,
        the references of its retriever and the run's ``context``.
        """
        parts = [_render(text, variables, f"{self.name}: {part}") for part, text in self.prompt_parts().items()]
        return _join(*parts, self._inline_skills(self.skills))

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

        The model (``model``, else the definition's) runs on a copy of ``llm``, else on
        `ChatGwenlake`. What the code passes wins over the files; ``tools``, ``skills``, ``team``
        and ``mcp_servers`` are added to the harness ones.
        """
        backend = _backend(model or self.config.model, llm, self.name)
        settings = {"reasoning_effort": self.reasoning_effort} if self.reasoning_effort else {}
        _apply_settings(backend, {**settings, **self.config.model_settings, **(model_settings or {})}, self.name)

        params: Dict[str, Any] = {
            "name": self.name,
            "description": self.config.description,
            "instructions": self.instructions(variables) or None,
            "llm": backend,
            "tools": list(self.tools),
            "skills": self._loaded_skills(self.skills),
            "team": [self._build_subagent(sub, backend, variables) for sub in self.subagents] or None,
            "mcp_servers": [_mcp_server(server, self.directory) for server in self.mcp_servers],
        }
        if self.config.output_schema is not None:
            params["response_model"] = schema_model(self.config.output_schema)
        if self.config.max_turns is not None:
            params["max_turns"] = self.config.max_turns
        if self.config.long_term_memory is not None and "memory" not in agent_kwargs:
            params["memory"] = _long_term_memory(self.config.long_term_memory, variables)
        for key, value in agent_kwargs.items():
            params[key] = (params.get(key) or []) + list(value or []) if key in ADDED_FIELDS else value
        return Agent(**params)

    def _build_subagent(self, sub: Subagent, parent: ChatBase, variables: Optional[Dict[str, Any]]) -> Agent:
        """A team member, on a copy of the orchestrator's backend."""
        where = f"subagents/{sub.name}"
        backend = _backend(sub.config.model, parent, where)
        _apply_settings(backend, sub.config.model_settings, where)
        skills = [skill for skill in self.skills if skill.name in sub.config.skills]
        params: Dict[str, Any] = {}
        if sub.config.max_turns is not None:
            params["max_turns"] = sub.config.max_turns
        return Agent(
            name=sub.name,
            description=sub.config.description,
            instructions=_join(_render(sub.config.instructions or "", variables, where), self._inline_skills(skills))
            or None,
            llm=backend,
            tools=list(sub.tools),
            skills=self._loaded_skills(skills),
            **params,
        )

    def _inline_skills(self, skills: List[Skill]) -> str:
        """The full text of the skills, for the prompt."""
        if not skills or not self.skills_inline:
            return ""
        blocks = (f'<skill name="{s.name}">\n{s.description}\n\n{s.content.strip()}\n</skill>' for s in skills)
        return "## Skills\n\nApply the skill whose situation matches.\n\n" + "\n\n".join(blocks)

    def _loaded_skills(self, skills: List[Skill]) -> List[Skill]:
        """The skills the agent loads on demand: none when they are already in the prompt."""
        return [] if self.skills_inline else list(skills)


# ---------------------------------------------------------------------- public helpers


def schema_model(schema: Dict[str, Any]) -> Type[BaseModel]:
    """A `response_model` from a JSON Schema.

    The schema is sent to the model as is; the answer comes back with its fields as attributes
    (``.model_dump()`` gives the dict), not validated against the schema.
    """
    frozen = copy.deepcopy(schema)

    class SchemaModel(BaseModel):
        model_config = ConfigDict(extra="allow")

        @classmethod
        def model_json_schema(cls, *args: Any, **kwargs: Any) -> Dict[str, Any]:
            return copy.deepcopy(frozen)  # callers make it strict in place

    SchemaModel.__name__ = re.sub(r"[^A-Za-z0-9_-]", "_", str(schema.get("title") or "Output"))
    return SchemaModel


def builtin_tool(name: str, kwargs: Dict[str, Any], where: str) -> BaseTool:
    """A tool of `gwenflow.tools`, by class name."""
    import gwenflow.tools as builtins

    if name in ("BaseTool", "Tool") or name.startswith("MCP") or name not in builtins.__all__:
        raise HarnessError(f"{where}: unknown capability or tool '{name}'")
    try:
        return getattr(builtins, name)(**kwargs)
    except Exception as e:
        raise HarnessError(f"{where}: cannot create {name}: {type(e).__name__}: {e}") from e


# ---------------------------------------------------------------------- reading the files


def _read_data(path: Path) -> Any:
    """The content of a YAML or JSON file."""
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text) if path.suffix == ".json" else (yaml.safe_load(text) or {})
    except (json.JSONDecodeError, yaml.YAMLError) as e:
        raise HarnessError(f"{path.name}: invalid file: {e}") from e


def _read_markdown(path: Path) -> str:
    return _strip_comments(path.read_text(encoding="utf-8"))


def _strip_comments(text: str) -> str:
    return _COMMENT.sub("", text).strip()


def _parse(model: Type[BaseModel], raw: Any, where: str) -> Any:
    """``raw`` as a ``model``, or a `HarnessError` that says where and what is wrong."""
    if not isinstance(raw, dict):
        raise HarnessError(f"{where}: expected a mapping")
    try:
        return model.model_validate(raw)
    except ValidationError as e:
        errors = "; ".join(f"{'.'.join(str(p) for p in x['loc']) or 'value'}: {x['msg']}" for x in e.errors())
        raise HarnessError(f"{where}: {errors}") from e


def _set_from_file(config: HarnessConfig, key: str, value: Any, source: str) -> None:
    """A value read from its own file, refused if the definition already holds it."""
    if not value:
        return
    if getattr(config, key):
        raise HarnessError(f"`{key}` is written in both {CONFIG_FILE} and {source}: keep one")
    setattr(config, key, value)


def _capability(entry: Capability, where: str) -> Tuple[str, Any]:
    """The name and value of a capability entry: ``Name``, ``{Name: value}`` or ``{Name: {...}}``."""
    if isinstance(entry, str):
        return entry, None
    if isinstance(entry, dict) and len(entry) == 1:
        ((name, value),) = entry.items()
        return name, value
    raise HarnessError(f"{where}: a capability is `Name` or `{{Name: ...}}`, got {entry!r}")


def _kwargs(value: Any, positional: Optional[str], where: str) -> Dict[str, Any]:
    """The keyword arguments of a capability: a single value stands for its ``positional`` argument."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if positional is None:
        raise HarnessError(f"{where}: a tool takes keyword arguments, got {value!r}")
    return {positional: value}


def _unique(tools: List[BaseTool], where: str) -> List[BaseTool]:
    names = [tool.name for tool in tools]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise HarnessError(f"{where}: several tools are named {duplicates}")
    return tools


def _read_skills(skills_folder: Optional[Path], base: Path, options: SkillsCapability) -> List[Skill]:
    """The skills of ``skills/`` and of ``options.directories`` (relative to ``base``), filtered by name."""
    folders = [skills_folder] if skills_folder is not None and skills_folder.is_dir() else []
    for entry in options.directories:
        folder = (base / entry).resolve()
        if not folder.is_dir():
            raise HarnessError(f"Skills: directory '{entry}' not found")
        folders.append(folder)

    by_name: Dict[str, Skill] = {}
    for folder in folders:
        for skill in SkillsDirectory(str(folder)).skills:
            by_name.setdefault(skill.name, skill)

    skills = list(by_name.values())
    if options.include is not None:
        skills = [skill for skill in skills if skill.name in options.include]
    if options.exclude is not None:
        skills = [skill for skill in skills if skill.name not in options.exclude]
    return skills


def _read_subagents(folder: Path, harness_tools: List[BaseTool]) -> List[Subagent]:
    """``subagents/<name>/AGENT.md``: a definition in the frontmatter, the instructions in the body."""
    if not folder.is_dir():
        return []
    tools_by_name = {tool.name: tool for tool in harness_tools}
    subagents: List[Subagent] = []
    for agent_file in sorted(folder.glob("*/AGENT.md")):
        where = f"subagents/{agent_file.parent.name}/AGENT.md"
        frontmatter, body = parse_frontmatter(agent_file.read_text(encoding="utf-8"))
        config = _parse(SubagentConfig, frontmatter, where)
        if config.instructions and _strip_comments(body):
            raise HarnessError(f"{where}: `instructions` is written in both the frontmatter and the body: keep one")
        config.instructions = config.instructions or _strip_comments(body)
        tools = [tools_by_name[n] if n in tools_by_name else builtin_tool(n, {}, where) for n in config.tools]
        subagents.append(Subagent(name=config.name or agent_file.parent.name, config=config, tools=tools))
    return subagents


def _read_tools(folder: Path) -> List[BaseTool]:
    """``tools/*.py``: a module-level ``tools`` list, else every public annotated and documented function."""
    if not folder.is_dir():
        return []
    package = "gwenflow_harness_" + hashlib.sha256(str(folder).encode()).hexdigest()[:12]
    tools: List[BaseTool] = []
    for script in sorted(folder.glob("*.py")):
        if script.name.startswith("_"):
            continue
        module = _import(f"{package}.{script.stem}", script)
        if isinstance(getattr(module, "tools", None), list):
            tools.extend(module.tools)
            continue
        for fn_name, fn in inspect.getmembers(module, inspect.isfunction):
            if not fn_name.startswith("_") and fn.__module__ == module.__name__ and fn.__doc__ and fn.__annotations__:
                tools.append(Tool(fn))
    return tools


def _import(module_name: str, script: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module  # so its functions can be pickled and introspected
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        sys.modules.pop(module_name, None)
        raise HarnessError(f"tools/{script.name}: {type(e).__name__}: {e}") from e
    return module


# ---------------------------------------------------------------------- building the agent


def _prompt_text(value: Union[str, List[str], None]) -> str:
    """A prompt part as text: a list is paragraphs, comments are removed."""
    if isinstance(value, list):
        value = "\n\n".join(str(item).strip() for item in value)
    return _strip_comments(value or "")


def _join(*parts: str) -> str:
    return "\n\n".join(part for part in parts if part)


def _render(text: str, variables: Optional[Dict[str, Any]], where: str) -> str:
    """``text`` with its ``{{ name }}`` replaced by ``variables``; every name must have a value."""
    now = datetime.now(timezone.utc)
    values = {"date": now.strftime("%Y-%m-%d"), "datetime": now.isoformat(timespec="seconds"), **(variables or {})}
    missing = sorted({name for name in _VARIABLE.findall(text) if name not in values})
    if missing:
        raise HarnessError(f"{where} uses {missing}: pass them in `variables`")
    return _VARIABLE.sub(lambda match: str(values[match.group(1)]), text)


def _backend(model: Optional[str], base: Optional[ChatBase], where: str) -> ChatBase:
    """A copy of ``base`` running ``model``; without ``base``, a `ChatGwenlake`."""
    if base is None:
        if model is None:
            raise HarnessError(f"'{where}' has no model: set `model`, or pass `model=` or `llm=`")
        return ChatGwenlake(model=model)
    backend = copy.copy(base)
    if model is not None:
        backend.model = model
    return backend


def _apply_settings(backend: ChatBase, settings: Dict[str, Any], where: str) -> None:
    allowed = {f.name for f in dataclasses.fields(backend)} - {"model", "tools", "tool_choice", "response_format"}
    unknown = sorted(set(settings) - allowed)
    if unknown:
        raise HarnessError(f"{where}: {type(backend).__name__} has no model setting {unknown}")
    for key, value in settings.items():
        setattr(backend, key, value)


def _from_environment(value: Any) -> Any:
    """``value`` with every ``${NAME}`` replaced by the environment variable ``NAME``."""

    def lookup(match: re.Match) -> str:
        name = match.group(1)
        if name not in os.environ:
            raise HarnessError(f"environment variable {name} is not set")
        return os.environ[name]

    if isinstance(value, str):
        return _ENV_VARIABLE.sub(lookup, value)
    if isinstance(value, list):
        return [_from_environment(item) for item in value]
    if isinstance(value, dict):
        return {key: _from_environment(item) for key, item in value.items()}
    return value


def _mcp_server(server: McpCapability, directory: Path) -> Any:
    from gwenflow.tools.mcp import MCPServerSse, MCPServerStdio

    if server.url is not None:
        params: Dict[str, Any] = {"url": _from_environment(server.url)}
        if server.headers:
            params["headers"] = _from_environment(server.headers)
        return MCPServerSse(params, name=server.id)
    params = {"command": server.command, "args": _from_environment(server.args)}
    if server.env:
        params["env"] = _from_environment(server.env)
    if server.cwd is not None:
        params["cwd"] = str((directory / server.cwd).resolve())
    return MCPServerStdio(params, name=server.id)


def _long_term_memory(config: LongTermMemoryConfig, variables: Optional[Dict[str, Any]]) -> Any:
    from gwenflow.memory import MemoryToolset
    from gwenflow.stores.files import LocalFileMemoryStore, S3MemoryStore

    if config.store == "local":
        store = LocalFileMemoryStore(_render(config.path, variables, "long_term_memory"))
    else:
        store = S3MemoryStore(config.bucket, _render(config.prefix, variables, "long_term_memory"))
    return MemoryToolset(store, always=config.always, browse=config.browse, tools=config.tools)
