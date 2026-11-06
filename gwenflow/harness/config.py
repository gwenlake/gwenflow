"""The settings of a harness.

``harness.yaml``, the frontmatter of ``subagents/<name>/AGENT.md`` and the frontmatter of a
single-file agent (``Agent.from_file``).

Unknown keys are refused, so a typo fails at load time instead of being ignored.
"""

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SKILLS_INLINE = "inline"
SKILLS_TOOLS = "tools"

ToolEntry = Union[str, Dict[str, Dict[str, Any]]]
"""A built-in tool of `gwenflow.tools` by class name (``PubMedTool``), or ``{PubMedTool: {max_results: 5}}``."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class McpServerConfig(_Strict):
    """One MCP server: ``url`` (HTTP + SSE) or ``command`` (stdio).

    ``${NAME}`` in ``url``, ``headers``, ``args`` and ``env`` is replaced by the environment
    variable ``NAME`` when the agent is built, so tokens stay out of the files.
    """

    name: Optional[str] = None
    url: Optional[str] = None
    headers: Dict[str, str] = Field(default_factory=dict)
    timeout: Optional[float] = None
    sse_read_timeout: Optional[float] = None
    command: Optional[str] = None
    args: List[str] = Field(default_factory=list)
    env: Dict[str, str] = Field(default_factory=dict)
    cwd: Optional[str] = None

    @model_validator(mode="after")
    def _one_transport(self) -> "McpServerConfig":
        if (self.url is None) == (self.command is None):
            raise ValueError("an MCP server has either `url` or `command`")
        return self


class _Settings(_Strict):
    model: Optional[str] = None
    """The model name, on the backend passed in code (``llm=``), else on Gwenlake. A subagent's model runs on
    a copy of the orchestrator's backend."""
    model_settings: Dict[str, Any] = Field(default_factory=dict)
    """Set on the chat backend: ``temperature``, ``max_tokens``, ``reasoning_effort``, ..."""
    max_turns: Optional[int] = None


class HarnessConfig(_Settings):
    """``harness.yaml``, or the frontmatter of a single-file agent. Every key is optional."""

    name: Optional[str] = None
    """Defaults to the directory name (the file name without extension for a single-file agent)."""
    version: str = "0"
    description: Optional[str] = None
    tools: List[ToolEntry] = Field(default_factory=list)
    """Built-in tools, added to those of ``tools/*.py``."""
    approval: List[str] = Field(default_factory=list)
    """Names of the tools the agent runs only after the user approved the call."""
    mcp_servers: List[McpServerConfig] = Field(default_factory=list)
    skills: Literal["inline", "tools"] = SKILLS_TOOLS
    """``tools``: the agent loads a skill on demand (``load_skill``); ``inline``: the full text of every
    skill goes into the system prompt and no tool is added."""
    metadata: Dict[str, Any] = Field(default_factory=dict)
    """Free keys for the application, never read by gwenflow."""

    @field_validator("version", mode="before")
    @classmethod
    def _version_as_text(cls, value: Any) -> str:
        return str(value)


class SubagentConfig(_Settings):
    """The frontmatter of ``subagents/<name>/AGENT.md``; the markdown body is the subagent's system prompt."""

    name: Optional[str] = None
    """Defaults to the directory name. The orchestrator gets an ``ask_<name>`` tool."""
    description: str
    """What the orchestrator reads to decide to delegate."""
    tools: List[str] = Field(default_factory=list)
    """Names of the tools the subagent may use: harness tools, or built-in tools of `gwenflow.tools`."""
    skills: List[str] = Field(default_factory=list)
    """Names of the harness skills the subagent may load."""
    mode: Optional[str] = None
    """Accepted for compatibility with other agent formats (``mode: subagent``); ignored."""


def harness_json_schema() -> Dict[str, Any]:
    """The JSON Schema of ``harness.yaml``, for editor completion and checks."""
    return HarnessConfig.model_json_schema()
