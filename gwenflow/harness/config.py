"""The definition of an agent, with the key names of the Pydantic AI agent spec.

``harness.yaml`` in a harness directory, an agent file (``agent.yaml``, ``.json``, ``.md``
frontmatter) and the frontmatter of ``subagents/<name>/AGENT.md``. Shared with Pydantic AI:
``model``, ``name``, ``description``, ``instructions``, ``model_settings``, ``capabilities``,
``output_schema``, ``metadata``. Added by gwenflow: ``version``, ``max_turns``, ``guardrails``,
``knowledge``, ``tool_guidance``, ``long_term_memory``. Unknown keys are refused.
"""

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from gwenflow.exceptions import UserError

Capability = Union[str, Dict[str, Any]]
"""``Name``, ``{Name: value}`` or ``{Name: {key: value}}``, as in Pydantic AI."""


class HarnessError(UserError):
    """A harness that cannot be loaded or built."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SkillsCapability(_Strict):
    """``- Skills: {directories: shared/skills, include: [code-review]}``.

    ``skills/`` of a harness is always read; ``directories`` (relative to the harness) add others.
    """

    directories: List[str] = Field(default_factory=list)
    include: Optional[List[str]] = None
    exclude: Optional[List[str]] = None
    mode: Literal["inline", "tools"] = "tools"
    """gwenflow: ``tools`` loads a skill on demand; ``inline`` puts every skill in the prompt."""

    @field_validator("directories", mode="before")
    @classmethod
    def _one_or_many(cls, value: Any) -> Any:
        return [value] if isinstance(value, str) else value


class McpCapability(_Strict):
    """``- MCP: https://mcp.example.com/sse``, or ``{url, id, headers}``, or ``{command, args, env, cwd}``.

    ``${NAME}`` in the values is read from the environment, so tokens stay out of the files.
    """

    url: Optional[str] = None
    id: Optional[str] = None
    headers: Dict[str, str] = Field(default_factory=dict)
    command: Optional[str] = None
    args: List[str] = Field(default_factory=list)
    env: Dict[str, str] = Field(default_factory=dict)
    cwd: Optional[str] = None

    @model_validator(mode="after")
    def _one_transport(self) -> "McpCapability":
        if (self.url is None) == (self.command is None):
            raise ValueError("an MCP server has either `url` or `command`")
        return self


class LongTermMemoryConfig(_Strict):
    """The memory the agent writes itself across conversations (`MemoryToolset`).

    Not to be confused with ``knowledge``, fixed facts written by the authors of the harness.
    ``{{ name }}`` in ``path`` and ``prefix`` is filled from ``variables`` (``.memories/{{ user_id }}``).
    """

    store: Literal["local", "s3"] = "local"
    path: Optional[str] = None
    bucket: Optional[str] = None
    prefix: str = "memories"
    always: List[str] = Field(default_factory=list)
    browse: List[str] = Field(default_factory=list)
    tools: Optional[List[str]] = None

    @model_validator(mode="after")
    def _store_target(self) -> "LongTermMemoryConfig":
        if self.store == "local" and not self.path:
            raise ValueError("a local long-term memory needs a `path`")
        if self.store == "s3" and not self.bucket:
            raise ValueError("an s3 long-term memory needs a `bucket`")
        return self


class _Settings(_Strict):
    model: Optional[str] = None
    """The model name. It runs on the `llm` given in code, else on Gwenlake."""
    model_settings: Dict[str, Any] = Field(default_factory=dict)
    """Set on the chat backend: ``temperature``, ``max_tokens``, ``reasoning_effort``, ..."""
    max_turns: Optional[int] = None


class HarnessConfig(_Settings):
    """The definition of an agent; every key is optional.

    In a directory, each part of the prompt is written here or in its file (``system_prompt.md``,
    ``guardrails.md``, ``knowledge.md``, ``tool_guidance.md``), not both; same for
    ``output_schema`` (``output_schema.json``). The parts are composed in this order.
    """

    name: Optional[str] = None
    version: str = "0"
    description: Optional[str] = None
    instructions: Optional[Union[str, List[str]]] = None
    """Role, method, output format (``system_prompt.md``)."""
    guardrails: Optional[str] = None
    """Rules that always apply (``guardrails.md``)."""
    knowledge: Optional[str] = None
    """Lasting facts of the domain, written by the authors (``knowledge.md``), never changed by the model."""
    tool_guidance: Optional[str] = None
    """How to use the tools and the subagents (``tool_guidance.md``)."""
    capabilities: List[Capability] = Field(default_factory=list)
    """``Skills``, ``MCP``, ``Thinking``, ``WebSearch``, ``WebFetch``, and the tools of `gwenflow.tools`
    by class name (``- PubMedTool: {max_results: 5, requires_approval: true}``)."""
    output_schema: Optional[Dict[str, Any]] = None
    """JSON Schema of the answer (``output_schema.json``)."""
    long_term_memory: Optional[LongTermMemoryConfig] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    """Free keys for the application, never read by gwenflow."""

    @field_validator("version", mode="before")
    @classmethod
    def _version_as_text(cls, value: Any) -> str:
        return str(value)


class SubagentConfig(_Settings):
    """The frontmatter of ``subagents/<name>/AGENT.md``; the body is its ``instructions``."""

    name: Optional[str] = None
    description: str
    """What the orchestrator reads to decide to delegate."""
    instructions: Optional[str] = None
    tools: List[str] = Field(default_factory=list)
    """Names of harness tools, or tools of `gwenflow.tools`."""
    skills: List[str] = Field(default_factory=list)
    mode: Optional[str] = None
    """Accepted (``mode: subagent``) and ignored."""


def harness_json_schema() -> Dict[str, Any]:
    """The JSON Schema of ``harness.yaml``, for editor completion."""
    return HarnessConfig.model_json_schema()
