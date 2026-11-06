import asyncio
import contextvars
import copy
import json
import math
import queue
import re
import threading
import time
import warnings
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Type,
    Union,
)

from pydantic import BaseModel, ValidationError

from gwenflow.agents.subagents import (
    FAST_MODEL,
    SPAWN_AGENT_TOOL,
    SUBAGENT_ABANDONED,
    SUBAGENT_EVENT_POLL_SECONDS,
    SUBAGENT_INSTRUCTIONS,
    SUBAGENT_MODELS,
    SUBAGENT_NAME,
    SUBAGENT_NAME_MAX_CHARS,
    spawn_agent_description,
    spawn_agent_parameters,
)
from gwenflow.llms import ChatBase, ChatOpenAI
from gwenflow.llms.base import CONTEXT_SAFETY_RATIO
from gwenflow.logger import logger
from gwenflow.memory import ChatMemoryBuffer, ChatSummaryMemoryBuffer, MemoryToolset
from gwenflow.memory.chat_memory_buffer import DEFAULT_MAX_FRACTION
from gwenflow.memory.compaction import PINNED_KEY, pin
from gwenflow.retriever import Retriever
from gwenflow.skills import Skill, SkillsToolset
from gwenflow.telemetry import tracer
from gwenflow.telemetry._semconv import (
    DELEGATION_FINISH_REASON,
    SUBAGENT_DESCRIPTION,
    SUBAGENT_MODEL,
    SUBAGENT_MODEL_NAME,
    SUBAGENT_SKILLS,
    SUBAGENT_STRUCTURED,
    SUBAGENT_TOOLS,
)
from gwenflow.tools import BaseTool, MCPServer, Tool
from gwenflow.types import (
    AgentEventCompleted,
    AgentEventContent,
    AgentEventStarted,
    AgentEventSubagentCompleted,
    AgentEventSubagentStarted,
    AgentEventThinking,
    AgentEventToolApprovalRequired,
    AgentEventToolCompleted,
    AgentEventToolStarted,
    AgentResponse,
    AgentResponseEvent,
    AgentUsage,
    ContextUsage,
    ContextWindowInfo,
    Message,
    ModelResponse,
    RequestUsage,
    TextContent,
    ToolApprovalRequest,
    ToolCall,
    ToolDenied,
    ToolResponse,
)
from gwenflow.utils import extract_json_str
from gwenflow.utils.durable import in_workflow, new_id
from gwenflow.utils.tokens import keep_tokens_from_text

try:
    from temporalio.exceptions import ActivityError
except ImportError:  # temporal is optional

    class ActivityError(Exception):  # type: ignore[no-redef]
        pass


_CURRENT_TOOL_CALL_ID: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "gwenflow_current_tool_call_id", default=None
)

FINAL_RESULT_TOOL = "final_result"
READ_TOOL_OUTPUT = "read_tool_output"
MAX_STORED_TOOL_OUTPUTS = 50

PROMPT_KNOWLEDGE = """\
## Use the following references from the knowledge base if it helps:

<references>
{references}
</references>
"""

PROMPT_CONTEXT = """\
## Use the following context if it helps:

<context>
{context}
</context>
"""

PROMPT_JSON_SCHEMA = """\
## Provide your output using the following JSON schema:

<json_schema>
{json_schema}
</json_schema>
"""


@dataclass(kw_only=True)
class FinalResultTool(BaseTool):
    """The `response_model` exposed as a tool (pydantic-ai's `final_result`).

    When an agent has both tools and a `response_model`, the schema is offered
    to the model as one more function. Calling it *is* the final answer: the
    agent validates the arguments and ends the run instead of executing
    anything. Tool-capable models end their tool loop with a tool call far
    more reliably than they switch to writing raw JSON text, and no native
    `response_format` is needed while tools are on offer (see
    `Agent._configure_llm_for_turn`).
    """

    name: str = FINAL_RESULT_TOOL
    description: str = (
        "Return your final answer. Call this once you have gathered everything you need: "
        "its arguments are the complete, final answer."
    )

    def _run(self, **kwargs: Any) -> Any:  # never executed: see Agent._answer_from_tool_calls
        return kwargs


MAX_VALIDATION_RETRIES = 3
"""How many times an invalid structured (`response_model`) answer is sent
back to the model for correction before giving up (parsed=None, raw
content kept)."""

ToolApprovals = Dict[str, Union[bool, ToolDenied]]
"""The user's decisions on the tool calls a run paused on, by tool call id:
True runs the call, False or a `ToolDenied` (with the reason the model reads)
refuses it. Only an explicit True approves."""


class _SubagentPausedError(Exception):
    pass


@dataclass
class _PausedSubagent:
    worker: "Agent"
    description: str
    requests: List[ToolApprovalRequest]
    elapsed: float


UNANSWERED_APPROVAL = "The user did not answer this approval request and sent a new message instead."
"""Tool result of a pending call left without a decision when the next run starts."""


@dataclass
class Agent:
    id: str | None = None
    """The unique id of the agent."""

    name: str | None = None
    """The name of the agent."""

    description: str | None = None
    """A description of the agent, used as a handoff, so that the manager knows what it does."""

    system_prompt: str | None = None
    """System prompt."""

    instructions: str | List[str] | None = None
    """The instructions for the agent."""

    response_model: Optional[Union[Dict[str, Any], Type[BaseModel]]] = None
    """Response model."""

    llm: Optional[ChatBase] = None
    """The model implementation to use when invoking the LLM."""

    tools: List[BaseTool] = field(default_factory=list)
    """A list of tools that the agent can use."""

    mcp_servers: List[MCPServer] = field(default_factory=list)
    """A list of MCP servers that the agent can use."""

    tool_choice: Literal["auto", "required", "none"] | str | None = None
    """The tool choice to use when calling the model."""

    reasoning_model: Optional[ChatBase] = None
    """Reasoning model."""

    history: ChatMemoryBuffer | None = None
    """Conversation memory. None: a `ChatSummaryMemoryBuffer` on `max_fraction`
    0.85 of the model's window, summarizing with `summarizer` or a copy of `llm`."""

    summarizer: Optional[ChatBase] = None
    """LLM used to summarize the history when it is compacted; a cheaper model
    than `llm` is a good choice. None: a copy of `llm` without tools. Ignored
    when `history` is given."""
    """Historical messages for the agent."""

    retriever: Optional[Retriever] = None
    """Retriever for the agent."""

    team: List["Agent"] | None = None
    """Team of agents. Each member is exposed to the model as an
    `ask_<name>` tool. Every handoff runs on a fresh clone of the member
    (same configuration, empty history): delegated tasks must therefore be
    self-contained — the member never sees the parent conversation."""

    subagents: bool = False

    subagent_llm: Optional[ChatBase] = None

    subagent_max_turns: Optional[int] = 25

    subagent_max_concurrency: Optional[int] = 4

    subagent_timeout: Optional[float] = None

    max_turns: Optional[int] = 100
    """Maximum turn (tool calls, llm calls) an agent can do."""

    commands: Any = None
    """Deprecated and ignored: slash commands were removed in 1.6.0."""

    warn_near_limits: bool = False
    """Tell the model, in the system prompt of the request, when the run is
    close to a limit: `warning_threshold` of the context budget or of
    `max_turns` (URGENT), compaction imminent or `critical_remaining_turns`
    left (CRITICAL). Request-scoped: nothing is written in the history."""

    warning_threshold: float = 0.7
    critical_remaining_turns: int = 3

    on_context_usage: Optional[Callable[[ContextUsage], None]] = None
    """Called before every request with the size of the history against the
    model's window (see `ContextUsage`)."""

    max_tool_output_tokens: Optional[int] = None
    """Cap on a single tool result. A longer result is stored aside: the model
    gets its beginning plus a reference, and a `read_tool_output` tool to page
    through the rest — instead of one grep or file read filling the context.
    None (default) keeps results whole."""

    _tool_outputs: Dict[str, str] = field(default_factory=dict, init=False, repr=False)
    """Full tool results set aside by `max_tool_output_tokens`, by reference.
    Shared with clones, so a handoff's references stay readable."""

    skills: List[Skill] = field(default_factory=list)

    memory: Optional[MemoryToolset] = None
    """Long-term markdown memory: a `## Memory` block in the system prompt plus
    the editing tools (`read_memory`, `memory_edit`, ...). Load it with
    `await memory.aload()` before the run and persist with `await memory.aflush()`
    after; the block is not rendered when `system_prompt` is set."""
    """Skills that extend the agent's instructions."""

    total_usage: AgentUsage = field(default_factory=AgentUsage)
    """Cumulative usage across every run of this agent, team delegations
    included."""

    pricing: Optional[Dict[str, float]] = None
    """Optional price grid in dollars per MILLION tokens, e.g.
    {"input": 3.0, "output": 15.0} (optional keys: "cache_read",
    "cache_write"). Not used by gwenflow itself since slash commands were
    removed; kept for callers that estimate costs from `total_usage`."""

    delegation_usage: AgentUsage = field(default_factory=AgentUsage)
    """Cumulative usage of team handoffs (sub-agent runs). Informational
    breakdown: this spend is already folded into `total_usage`."""

    _pending_delegation_usage: AgentUsage = field(default_factory=AgentUsage, init=False, repr=False)
    """Delegation usage accumulated since the last drain. Guarded by
    `_delegation_lock` because handoffs may run in parallel threads
    (`aexecute_tool_calls`); the run loops drain it after each tool round."""

    _delegation_lock: Any = field(default_factory=threading.Lock, init=False, repr=False)

    _handoff_tools: List[BaseTool] = field(default_factory=list, init=False, repr=False)
    """The subset of `tools` that wraps team members, kept separate so
    `clone()` can rebuild them bound to the clone."""

    _spawn_tools: List[BaseTool] = field(default_factory=list, init=False, repr=False)

    _subagent_slots: Optional[threading.BoundedSemaphore] = field(default=None, init=False, repr=False)

    _stop_requested: threading.Event = field(default_factory=threading.Event, init=False, repr=False)

    _paused_subagents: Dict[str, _PausedSubagent] = field(default_factory=dict, init=False, repr=False)

    _subagent_approvals: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    _subagent_events: Optional[queue.SimpleQueue] = field(default=None, init=False, repr=False)

    _stream_subagent_events: bool = field(default=False, init=False, repr=False)

    _subagent_run_id: Optional[str] = field(default=None, init=False, repr=False)

    durability: Optional[Any] = None
    """Durable execution backend, e.g. `TemporalDurability()`. None: plain in-process runs.
    Last field on purpose: existing positional arguments keep their meaning."""

    def __post_init__(self) -> None:
        if self.id is None:
            self.id = new_id()
        if self.commands is not None:
            warnings.warn(
                "Agent(commands=...) is deprecated and ignored: slash commands were removed in gwenflow 1.6.0.",
                DeprecationWarning,
                stacklevel=3,
            )
        if self.llm is None:
            self.llm = ChatOpenAI()
        else:
            self.llm = copy.copy(self.llm)
        if self.history is None:
            summarizer = copy.copy(self.summarizer if self.summarizer is not None else self.llm)
            summarizer.tools = []
            summarizer.tool_choice = None
            summarizer.response_format = None
            self.history = ChatSummaryMemoryBuffer(
                max_fraction=DEFAULT_MAX_FRACTION,
                llm=summarizer,
            )
            self._sync_context_window()
        if self.response_model:
            self.llm.response_format = self.response_model
        if self.skills:
            self.tools = list(self.tools) + SkillsToolset(self.skills).get_tools()
            self.tools = list(self.tools) + self._skill_script_tools()
        if self.max_tool_output_tokens and not any(t.name == READ_TOOL_OUTPUT for t in self.tools):
            self.tools = list(self.tools) + [self._tool_output_reader()]
        if self.memory:
            self.tools = list(self.tools) + self.memory.get_tools()
        if self.team:
            self._handoff_tools = self._build_handoff_tools()
            self.tools = list(self.tools) + self._handoff_tools
        if self.subagents:
            self._check_subagent_limits()
            self._subagent_slots = self._new_subagent_slots()
            self._spawn_tools = self._build_spawn_tools()
            self.tools = list(self.tools) + self._spawn_tools
        if self.tools or self.mcp_servers:
            self.llm.tools = self._llm_tools()
            self.llm.tool_choice = self.tool_choice
        else:
            self.llm.tools = None
            self.llm.tool_choice = None
        if self.durability is not None:
            self.durability.bind(self)

    def _build_handoff_tools(self) -> List[BaseTool]:
        """Expose each team member as a callable tool the orchestrator can invoke.

        Each handoff runs on a fresh `clone()` of the member: concurrent
        delegations in one turn (parallelized by `aexecute_tool_calls`) never
        share history or per-run state, and every task starts from a clean
        slate. The clone's spend (LLM calls and compaction) is recorded on
        this agent and folded into the current run's usage by the run loops.
        """

        def make_handoff(member: "Agent") -> Callable[[str], str]:
            def _handoff(task: str) -> str:
                """Send a task to a teammate agent and return its final answer.

                Args:
                    task: The full task or question to delegate to the teammate.
                """
                worker = member.clone()
                response = worker.run(task)
                self._record_delegation_usage(worker, response)
                return self._delegation_result("handoff", member.name or "agent", response)

            return _handoff

        tools: List[BaseTool] = []
        used_names: set[str] = set()
        for member in self.team or []:
            slug = re.sub(r"\W+", "_", (member.name or "agent").strip().lower()).strip("_") or "agent"
            tool_name = f"ask_{slug}"
            n = 2
            while tool_name in used_names:
                tool_name = f"ask_{slug}_{n}"
                n += 1
            used_names.add(tool_name)

            desc = member.description or f"Delegate a task to the '{member.name or slug}' agent."
            tools.append(Tool(make_handoff(member), name=tool_name, description=desc))
        return tools

    @staticmethod
    def _delegation_result(kind: str, agent_name: str, response: AgentResponse) -> str:
        tracer.set_attributes({DELEGATION_FINISH_REASON: response.finish_reason})
        if response.finish_reason == "max_turns":
            tracer.mark_error(message=f"{agent_name} reached max_turns", error_type="delegation_max_turns")
            return (
                f"[{kind} failed] The '{agent_name}' agent stopped "
                f"after reaching its max_turns limit without a final answer."
            )
        if response.finish_reason == "approval_required":
            tools = ", ".join(r.tool_name for r in response.pending_approvals)
            tracer.mark_error(
                message=f"{agent_name} needs approval to run {tools}", error_type="delegation_approval_required"
            )
            return (
                f"[{kind} failed] The '{agent_name}' agent needs the user's "
                f"approval to run {tools}, which a delegated task cannot ask for."
            )
        if isinstance(response.parsed, BaseModel):
            return response.parsed.model_dump_json()
        if isinstance(response.parsed, (dict, list)):
            return json.dumps(response.parsed, ensure_ascii=False)
        return response.content or ""

    def _build_spawn_tools(self) -> List[BaseTool]:
        def spawn_agent(
            description: str,
            prompt: str,
            tools: Optional[List[str]] = None,
            skills: Optional[List[str]] = None,
            model: str = FAST_MODEL,
            output_schema: Optional[Dict[str, Any]] = None,
        ) -> str:
            return self._spawn_subagent(description, prompt, tools or [], skills or [], model, output_schema)

        tool = Tool(spawn_agent, name=SPAWN_AGENT_TOOL, max_retries=1)
        self._describe_spawn_tool(tool)
        return [tool]

    def _describe_spawn_tool(self, tool: BaseTool) -> None:
        with_model = self.subagent_llm is not None
        skill_names = list(dict.fromkeys(s.name for s in self.skills))
        tool.description = spawn_agent_description(with_model, with_skills=bool(skill_names))
        tool.parameters = spawn_agent_parameters([t.name for t in self._delegable_tools()], skill_names, with_model)

    def _spawn_subagent(
        self,
        description: str,
        prompt: str,
        tool_names: List[str],
        skill_names: List[str],
        model: str,
        output_schema: Optional[Dict[str, Any]],
    ) -> str:
        call_id = _CURRENT_TOOL_CALL_ID.get()
        paused = self._paused_subagents.pop(call_id, None) if call_id else None
        if paused is not None:
            worker, description, elapsed = paused.worker, paused.description, paused.elapsed
            approvals = {
                r.tool_call_id: self._subagent_approvals[r.tool_call_id]
                for r in paused.requests
                if r.tool_call_id in self._subagent_approvals
            }
            response, elapsed = self._run_subagent(worker, None, approvals, call_id, elapsed)
        else:
            tracer.set_attributes(
                {
                    SUBAGENT_DESCRIPTION: description,
                    SUBAGENT_TOOLS: tool_names,
                    SUBAGENT_SKILLS: skill_names,
                    SUBAGENT_MODEL: model,
                    SUBAGENT_STRUCTURED: output_schema is not None,
                }
            )
            error = self._subagent_request_error(tool_names, skill_names, model, output_schema)
            if error:
                tracer.mark_error(message=error, error_type="invalid_subagent_request")
                return f"[sub-agent failed] {error}"
            available = {t.name: t for t in self._delegable_tools()}
            available_skills = {skill.name: skill for skill in self.skills}
            llm = self.subagent_llm if model == FAST_MODEL and self.subagent_llm is not None else self.llm
            tools = [available[name] for name in dict.fromkeys(tool_names)]
            skills = [available_skills[name] for name in dict.fromkeys(skill_names)]
            worker = self._build_subagent(description, tools, skills, llm, output_schema)
            tracer.set_attributes({SUBAGENT_MODEL_NAME: getattr(llm, "model", None)})
            logger.info(
                f"[Sub-agent] {description} (tools: {', '.join(t.name for t in tools) or 'none'}; "
                f"skills: {', '.join(s.name for s in skills) or 'none'})"
            )
            self._emit_subagent_event(
                AgentEventSubagentStarted(
                    agent_id=self.id,
                    run_id=self._subagent_run_id or "",
                    tool_call_id=call_id,
                    subagent_id=worker.id,
                    description=description,
                    tools=[t.name for t in tools],
                    skills=[s.name for s in skills],
                    model=getattr(llm, "model", None),
                )
            )
            response, elapsed = self._run_subagent(worker, prompt, None, call_id, 0.0)

        if response is None:
            reason = f"The sub-agent did not finish within {self.subagent_timeout:g}s and was stopped."
            tracer.set_attributes({DELEGATION_FINISH_REASON: "timeout"})
            tracer.mark_error(message=reason, error_type="delegation_timeout")
            self._emit_subagent_completed(call_id, worker, description, "timeout", None, elapsed)
            return f"[sub-agent failed] {reason} Do the task yourself, or split it into smaller tasks."
        if response.finish_reason == "approval_required" and call_id:
            requests = [replace(r, subagent_call_id=call_id) for r in response.pending_approvals]
            self._paused_subagents[call_id] = _PausedSubagent(worker, description, requests, elapsed)
            tracer.set_attributes({DELEGATION_FINISH_REASON: "approval_required"})
            raise _SubagentPausedError()
        self._emit_subagent_completed(call_id, worker, description, response.finish_reason, response.usage, elapsed)
        return self._delegation_result("sub-agent", worker.name, response)

    def _run_subagent(
        self,
        worker: "Agent",
        prompt: Optional[str],
        approvals: Optional[ToolApprovals],
        call_id: Optional[str],
        elapsed: float,
    ) -> tuple[Optional[AgentResponse], float]:
        def execute() -> AgentResponse:
            if not self._stream_subagent_events:
                return worker.run(prompt, approvals=approvals)
            response = None
            for item in worker.run_stream(prompt, approvals=approvals):
                if isinstance(item, AgentResponse):
                    response = item
                elif not worker._stop_requested.is_set():
                    item.metadata["subagent_call_id"] = call_id
                    self._emit_subagent_event(item)
            return response

        def work() -> AgentResponse:
            response = execute()
            self._record_delegation_usage(worker, response)
            return response

        if self._subagent_slots is not None:
            self._subagent_slots.acquire()
        started = time.monotonic()
        if self.subagent_timeout is None:
            try:
                return work(), elapsed + time.monotonic() - started
            finally:
                if self._subagent_slots is not None:
                    self._subagent_slots.release()

        done = threading.Event()
        outcome: Dict[str, Any] = {}

        def target() -> None:
            try:
                outcome["response"] = work()
            except BaseException as e:
                outcome["error"] = e
            finally:
                if self._subagent_slots is not None:
                    self._subagent_slots.release()
                done.set()

        threading.Thread(target=contextvars.copy_context().run, args=(target,), daemon=True).start()
        finished = done.wait(self.subagent_timeout)
        elapsed += time.monotonic() - started
        if not finished:
            worker._stop_requested.set()
            return None, elapsed
        if "error" in outcome:
            raise outcome["error"]
        return outcome["response"], elapsed

    def _emit_subagent_event(self, event: AgentResponseEvent) -> None:
        if self._subagent_events is not None:
            self._subagent_events.put(event)

    def _emit_subagent_completed(
        self,
        call_id: Optional[str],
        worker: "Agent",
        description: str,
        finish_reason: Optional[str],
        usage: Optional[AgentUsage],
        elapsed: float,
    ) -> None:
        self._emit_subagent_event(
            AgentEventSubagentCompleted(
                agent_id=self.id,
                run_id=self._subagent_run_id or "",
                tool_call_id=call_id,
                subagent_id=worker.id,
                description=description,
                finish_reason=finish_reason,
                usage=usage,
                duration_seconds=round(elapsed, 3),
            )
        )

    def _subagent_request_error(
        self,
        tool_names: List[str],
        skill_names: List[str],
        model: str,
        output_schema: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        delegable = [t.name for t in self._delegable_tools()]
        unknown = [name for name in tool_names if name not in delegable]
        if unknown:
            return f"Unknown or non-delegable tools: {', '.join(unknown)}. Delegable tools: {', '.join(delegable) or 'none'}."
        skills = [skill.name for skill in self.skills]
        unknown = [name for name in skill_names if name not in skills]
        if unknown:
            return f"Unknown skills: {', '.join(unknown)}. Available skills: {', '.join(skills) or 'none'}."
        if model not in SUBAGENT_MODELS:
            return f"Unknown model '{model}'. Use one of: {', '.join(SUBAGENT_MODELS)}."
        if output_schema is not None and not (
            isinstance(output_schema, dict) and ("type" in output_schema or "properties" in output_schema)
        ):
            return "output_schema must be a JSON Schema object, with a 'type' or 'properties' key."
        return None

    def _check_subagent_limits(self) -> None:
        value = self.subagent_max_concurrency
        if value is not None and value < 1:
            raise ValueError(f"`subagent_max_concurrency` must be at least 1, or None for no limit (got {value}).")
        if self.subagent_timeout is not None and self.subagent_timeout <= 0:
            raise ValueError(
                f"`subagent_timeout` must be positive, or None for no limit (got {self.subagent_timeout})."
            )

    def _new_subagent_slots(self) -> Optional[threading.BoundedSemaphore]:
        if not self.subagents or self.subagent_max_concurrency is None:
            return None
        return threading.BoundedSemaphore(self.subagent_max_concurrency)

    def _build_subagent(
        self,
        description: str,
        tools: List[BaseTool],
        skills: List[Skill],
        llm: ChatBase,
        output_schema: Optional[Dict[str, Any]] = None,
    ) -> "Agent":
        llm = copy.copy(llm)
        llm.tools = None
        llm.tool_choice = None
        llm.response_format = None
        label = " ".join(description.split())[:SUBAGENT_NAME_MAX_CHARS]
        return Agent(
            name=f"{SUBAGENT_NAME}: {label}" if label else SUBAGENT_NAME,
            instructions=list(SUBAGENT_INSTRUCTIONS),
            llm=llm,
            tools=tools,
            skills=skills,
            response_model=output_schema,
            max_turns=self.subagent_max_turns,
            max_tool_output_tokens=self.max_tool_output_tokens,
            pricing=self.pricing,
        )

    def _delegable_tools(self) -> List[BaseTool]:
        excluded = {SPAWN_AGENT_TOOL, READ_TOOL_OUTPUT}
        if self.skills:
            excluded |= {t.name for t in SkillsToolset(self.skills).get_tools()}
        if self.memory:
            excluded |= {t.name for t in self.memory.get_tools()}
        delegation_ids = {id(t) for t in self._handoff_tools + self._spawn_tools}
        return [t for t in self.get_all_tools() if t.name not in excluded and id(t) not in delegation_ids]

    def _skill_script_tools(self) -> List[BaseTool]:
        """Tools defined in the skills' `scripts/` files, deduplicated by name.

        A user passing `tools=sd.get_tools()` explicitly keeps working
        without duplicates.
        """
        known = {t.name for t in self.tools}
        tools: List[BaseTool] = []
        for skill in self.skills:
            for tool in skill.get_tools():
                if tool.name in known:
                    continue
                known.add(tool.name)
                tools.append(tool)
        return tools

    def clone(self) -> "Agent":
        """A new agent with this agent's configuration and a fresh, empty history.

        Configuration (prompts, tools, skills, team, retriever, pricing) is
        shared by reference; per-run mutable state is not: the clone gets its
        own history, its own llm copy (the run loop may downgrade
        `tool_choice` on it), fresh usage counters, and — when it has a team —
        handoff tools rebound to the clone so nested delegation spend is
        recorded on the right instance. Used by handoffs; also handy to run
        the same agent concurrently by hand.
        """
        worker = copy.copy(self)
        worker.id = new_id()
        worker.llm = copy.copy(self.llm)
        worker.history = self._fresh_history()
        worker.total_usage = AgentUsage()
        worker.delegation_usage = AgentUsage()
        worker._pending_delegation_usage = AgentUsage()
        worker._delegation_lock = threading.Lock()
        worker._stop_requested = threading.Event()
        if self.team or self.subagents:
            delegation_ids = {id(t) for t in self._handoff_tools + self._spawn_tools}
            base_tools = [t for t in self.tools if id(t) not in delegation_ids]
            worker._handoff_tools = worker._build_handoff_tools() if self.team else []
            worker._spawn_tools = []
            worker.tools = base_tools + worker._handoff_tools
            worker._subagent_slots = worker._new_subagent_slots()
            worker._paused_subagents = {}
            worker._subagent_approvals = {}
            worker._subagent_events = None
            worker._spawn_tools = worker._build_spawn_tools() if self.subagents else []
            worker.tools = worker.tools + worker._spawn_tools
            worker.llm.tools = worker.get_all_tools()
        return worker

    def _fresh_history(self) -> ChatMemoryBuffer:
        """A new, empty buffer with the same configuration as `self.history`."""
        kwargs: Dict[str, Any] = {
            "token_limit": self.history.token_limit,
            "max_fraction": self.history.max_fraction,
            "context_window": self.history.context_window,
            "context_window_resolved": self.history.context_window_resolved,
            "tokenizer_fn": self.history.tokenizer_fn,
        }
        summarizer = getattr(self.history, "llm", None)
        if summarizer is not None:
            kwargs["llm"] = summarizer
        return type(self.history)(**kwargs)

    def _record_delegation_usage(self, worker: "Agent", response: AgentResponse) -> None:
        """Fold a finished handoff's spend into this agent's accounting.

        Thread-safe: handoffs may run in parallel threads. The pending bucket
        is drained into the current run's usage after each tool round; the
        cumulative `delegation_usage` keeps the breakdown.
        """
        usage = AgentUsage()
        usage.add(response.usage)  # the worker's compaction spend is already inside
        with self._delegation_lock:
            self._pending_delegation_usage.add(usage)
            self.delegation_usage.add(usage)

    def _fold_delegation_usage(self, agent_response: AgentResponse) -> None:
        drained = self._drain_delegation_usage()
        agent_response.usage.add(drained)
        agent_response.delegation_usage.add(drained)

    def _finish_run(self, agent_response: AgentResponse, finish_reason: str) -> None:
        agent_response.finish_reason = finish_reason
        drain = getattr(self.history, "drain_usage", None)
        if drain is not None:
            agent_response.usage.add(drain())
        self.total_usage.add(agent_response.usage)

    def _interrupt_run(self, agent_response: AgentResponse) -> None:
        if agent_response.finish_reason is not None:
            return
        if self.team or self.subagents:
            self._fold_delegation_usage(agent_response)
        self._finish_run(agent_response, "interrupted")
        tracer.record_agent_result(agent_response)

    def _drain_delegation_usage(self) -> AgentUsage:
        """Take and reset the delegation usage accumulated by handoffs."""
        with self._delegation_lock:
            drained = self._pending_delegation_usage
            self._pending_delegation_usage = AgentUsage()
        return drained

    def tool(self, func: Optional[Callable] = None, *, requires_approval: bool = False) -> Callable:
        """Decorator that registers a function as a tool on this agent instance.

        Use it bare (`@agent.tool`) or with options (`@agent.tool(requires_approval=True)`).
        """

        def register(func: Callable) -> Callable:
            self.tools = list(self.tools) + [Tool(func, requires_approval=requires_approval)]
            self.llm.tools = self._llm_tools()
            self.llm.tool_choice = self.tool_choice
            return func

        if func is None:
            return register
        return register(func)

    def _format_context(self, context: Optional[Union[str, Dict[str, str]]]) -> str:
        text = ""
        if isinstance(context, str):
            text = f"<context>\n{context}\n</context>\n\n"
        elif isinstance(context, dict):
            for key in context.keys():
                text += f"<{key}>\n"
                text += context.get(key) + "\n"
                text += f"</{key}>\n\n"
        return text

    def _parse_tool_args(self, tool_call: ToolCall) -> Dict[str, Any]:
        """Best-effort parse of tool-call arguments for event payloads.

        Malformed JSON emitted by the model must not crash the run: the
        failure is surfaced to the model by `run_tool` itself, so here we
        degrade to a raw payload instead of raising.
        """
        args = tool_call.arguments
        if isinstance(args, dict):
            return args
        if not (isinstance(args, str) and args.strip()):
            return {}
        try:
            parsed = json.loads(args)
        except (json.JSONDecodeError, TypeError):
            return {"raw": args}
        return parsed if isinstance(parsed, dict) else {"raw": args}

    def _configure_llm_for_turn(self, structured: bool) -> None:
        """Decide whether this turn asks the provider for native structured output.

        Sending `response_format` (JSON schema) *and* `tools` in the same request
        works with OpenAI, but many OpenAI-compatible servers (vLLM, DeepSeek,
        Scaleway, Ollama...) apply the JSON grammar from the first token, which
        makes a tool call impossible: the model answers in JSON right away and
        hallucinates the data it should have fetched with its tools.

        So, when the agent has tools, `response_format` is withheld while tools
        are on offer. The schema is exposed as the `final_result` tool instead
        (see `FinalResultTool`) and is also in the system prompt, so the final
        answer normally arrives as a `final_result` call or as valid JSON
        text. If a text answer fails validation, the retry turn
        (`structured=True`) forces a `final_result` call through a named
        `tool_choice`: on vLLM/Ollama-style servers a forced tool call is
        honoured more reliably than a `response_format` grammar, and the
        arguments go through the same validation. `response_format` is never
        sent together with tools.

        Without tools the behaviour is unchanged: native structured output on
        every turn.
        """
        if self.response_model is None:
            return
        if not self.get_all_tools():
            self.llm.response_format = self.response_model
            return
        if structured:
            self.llm.response_format = None
            self.llm.tool_choice = {"type": "function", "function": {"name": FINAL_RESULT_TOOL}}
        else:
            self.llm.response_format = None
            self.llm.tool_choice = self.tool_choice

    def _final_result_tool(self) -> FinalResultTool:
        if isinstance(self.response_model, type) and issubclass(self.response_model, BaseModel):
            schema = self.response_model.model_json_schema()
        else:
            schema = dict(self.response_model)
        return FinalResultTool(parameters=schema)

    def _llm_tools(self) -> List[BaseTool]:
        """Tools to put on the wire.

        The agent's own tools, plus `final_result` when a `response_model`
        has to coexist with them (see `FinalResultTool`).
        """
        tools = self.get_all_tools()
        if tools and self.response_model is not None:
            tools = tools + [self._final_result_tool()]
        return tools

    def _answer_from_tool_calls(self, tool_calls: List[ToolCall]) -> Optional[str]:
        """Return the final JSON answer when the turn's tool call *is* the answer.

        Two cases qualify, both only with a `response_model`:

        - a call to `final_result` (the schema exposed as a tool);
        - a call to a tool that does not exist whose arguments validate
          against the schema: Qwen, Llama and friends sometimes "call" the
          schema by its own title (`Answer({...})`) instead of `final_result`.

        Like pydantic-ai, a valid final result ends the run even if the same
        turn carries other tool calls: those are not executed. Invalid
        arguments return None so that `run_tool` can feed the validation
        error back to the model as the tool result.

        Returns the canonical JSON text to record as the assistant's final
        content, or None when the turn is a normal tool turn.
        """
        if self.response_model is None or not tool_calls:
            return None
        known = {t.name for t in self.get_all_tools()}
        for tool_call in tool_calls:
            if tool_call.name in known:
                continue
            args = self._parse_tool_args(tool_call)
            if "raw" in args and len(args) == 1:
                continue
            is_valid, parsed, _ = self._validate_final_response(args)
            if not is_valid:
                continue
            skipped = [t.name for t in tool_calls if t is not tool_call]
            if skipped:
                logger.warning(f"Final result received: tool calls {skipped} not executed")
            if tool_call.name != FINAL_RESULT_TOOL:
                logger.debug(f"Tool call '{tool_call.name}' matches the response schema: treated as the final answer")
            if isinstance(parsed, BaseModel):
                return parsed.model_dump_json()
            return json.dumps(parsed)
        return None

    def _absorb_answer_tool_call(self, response: ModelResponse) -> None:
        """Rewrite in place a response whose tool call is really the final answer.

        See `_answer_from_tool_calls`. The tool calls are replaced by a text
        part so that the history records a plain assistant message (no
        orphan tool call waiting for a tool result) and the normal
        validation path runs.
        """
        answer = self._answer_from_tool_calls(response.tool_calls)
        if answer is None:
            return
        response.parts = [p for p in response.parts if not isinstance(p, ToolCall)] + [TextContent(content=answer)]

    def _validate_final_response(
        self, content_to_parse: str | Dict[str, Any] | List[Any] | None
    ) -> tuple[bool, Any, Optional[str]]:
        """Validate the final content against `response_model`.

        Single code path shared by run/arun/run_stream/arun_stream. Pydantic
        models are validated; plain dict schemas are parsed to JSON (fenced
        JSON accepted via `extract_json_str`) without schema validation; no
        `response_model` means passthrough.
        """
        if self.response_model is None:
            return True, content_to_parse, None

        is_pydantic_model = isinstance(self.response_model, type) and issubclass(self.response_model, BaseModel)

        try:
            if isinstance(content_to_parse, str):
                content_to_parse = json.loads(extract_json_str(content_to_parse))
            if is_pydantic_model:
                return True, self.response_model.model_validate(content_to_parse), None
            return True, content_to_parse, None
        except (ValidationError, json.JSONDecodeError, TypeError, ValueError) as e:
            error_msg = (
                f"Your final response failed validation against the required schema. "
                f"Error details:\n{str(e)}\n"
                f"Please correct the errors and return ONLY the valid JSON."
            )
            return False, None, error_msg

    def _tools_token_count(self) -> int:
        """Tokens the tool schemas cost on the wire.

        Every provider sends the same three fields per tool (name, description,
        JSON-schema parameters), so a single neutral estimate covers them all.
        These tokens never appear in the message list, so the memory buffer has
        to be told to reserve them — a large MCP server can otherwise eat the
        whole context margin without the buffer noticing.
        """
        tools = self.llm.tools or []
        tokenizer_fn = self.history.tokenizer_fn
        total = 0
        for tool in tools:
            schema = {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.parameters,
            }
            total += tokenizer_fn(json.dumps(schema, default=str))
        return total

    def _prepare_history(
        self,
        task: str,
        context: Optional[Union[str, Dict[str, str]]] = None,
    ) -> None:
        """Refresh the parts of the memory budget that are settled per run.

        The system prompt and the tool schemas sent alongside every request.
        """
        self._base_system_prompt = self.get_system_prompt(task=task, context=context)
        self.history.system_prompt = self._base_system_prompt
        for tool in self._spawn_tools:
            self._describe_spawn_tool(tool)
        self.history.reserved_tokens = self._tools_token_count()
        self._sync_context_window()

    _base_system_prompt: str = field(default="", init=False, repr=False)

    def _limit_warning(self, usage: ContextUsage, turns_left: int) -> Optional[str]:
        """The context notice for this request, or None when nothing is close."""
        if not self.warn_near_limits:
            return None
        lines: list[str] = []
        budget_fraction = usage.used_tokens / usage.limit_tokens if usage.limit_tokens else 0.0
        if budget_fraction >= 1.0:
            lines.append(
                f"[CRITICAL] The context budget is exhausted ({usage.fraction:.0%} of the window): older messages "
                "are being compacted. Conclude: give your final answer now."
            )
        elif budget_fraction >= self.warning_threshold:
            lines.append(
                f"[URGENT] {usage.fraction:.0%} of the context window is used; older messages will be compacted "
                "soon. Wrap up the task and give your final answer."
            )
        if self.max_turns:
            if turns_left <= self.critical_remaining_turns:
                lines.append(
                    f"[CRITICAL] {turns_left} turn(s) left before the run stops without an answer. "
                    "Give your final answer now."
                )
            elif 1 - turns_left / self.max_turns >= self.warning_threshold:
                lines.append(f"[URGENT] {turns_left} of {self.max_turns} turns left. Wrap up and answer.")
        if not lines:
            return None
        return "## Context notice\n\n" + "\n".join(lines)

    def _before_request(self, turns_left: int) -> None:
        """Per-request bookkeeping: window sync, usage report, limit notice."""
        if self.durability is not None and in_workflow():
            self.durability.before_model_request(self)
        self._sync_context_window()
        usage = self.history.context_usage()
        if self.on_context_usage is not None:
            self.on_context_usage(usage)
        warning = self._limit_warning(usage, turns_left)
        base = self._base_system_prompt or self.history.system_prompt or ""
        self.history.system_prompt = f"{base}\n\n{warning}".strip() if warning else base

    def _pin_task(self, messages: List[Message]) -> None:
        """Pin the task of this run so that no compaction removes it.

        The pin of the previous run's task is released: the first user message
        of the history is protected on its own, and the running summary
        covers the rest.
        """
        for message in self.history.messages:
            if message.extra.get("pinned_by") == "agent":
                message.extra.pop(PINNED_KEY, None)
                message.extra.pop("pinned_by", None)
        for message in reversed(messages):
            if message.role == "user":
                pin(message).extra["pinned_by"] = "agent"
                break

    def _sync_context_window(self) -> None:
        """Tell the history the context window of the LLM about to be called.

        Resolved on every request, so replacing `agent.llm` moves the budget.
        An LLM without `context_window_info` (a stub) is read through
        `get_context_size()`, which is `CONTEXT_SAFETY_RATIO` of its window.
        """
        info = None
        info_fn = getattr(self.llm, "context_window_info", None)
        if callable(info_fn):
            info = info_fn()
        if isinstance(info, ContextWindowInfo):
            window, resolved = info.tokens, info.resolved
        else:
            fraction = self.history.max_fraction or CONTEXT_SAFETY_RATIO
            window, resolved = math.ceil(self.llm.get_context_size() / fraction), True
        self.history.context_window = window
        self.history.context_window_resolved = resolved

    def context_usage(self) -> ContextUsage:
        """Size of the next request against the model's window (see `ContextUsage`)."""
        self._sync_context_window()
        return self.history.context_usage()

    def get_system_prompt(
        self,
        task: str,
        context: Optional[Union[str, Dict[str, str]]] = None,
    ) -> str:
        """Get the system prompt for the agent."""
        if self.system_prompt:
            return self.system_prompt.strip()

        system_prompt_parts = []

        if self.instructions:
            instructions = "## Instructions:\n\n"
            if isinstance(self.instructions, str):
                instructions += "- {instructions}".format(instructions=self.instructions)
            elif isinstance(self.instructions, list):
                instructions += "\n".join([f"- {i}" for i in self.instructions])
            system_prompt_parts.append(instructions)

        if self.skills:
            skills = "## Skills:\n\n"
            skills += SkillsToolset(self.skills).get_instructions()
            system_prompt_parts.append(skills)

        if self.response_model:
            if isinstance(self.response_model, type) and issubclass(self.response_model, BaseModel):
                schema_str = json.dumps(self.response_model.model_json_schema(), indent=4)
            else:
                schema_str = json.dumps(self.response_model, indent=4)
            prompt_schema = PROMPT_JSON_SCHEMA.format(json_schema=schema_str).strip()
            system_prompt_parts.append(prompt_schema)

        if self.memory:
            system_prompt_parts.append("## Memory:\n\n" + self.memory.get_instructions())

        if self.retriever:
            references = self.retriever.search(query=task)
            if len(references) > 0:
                references = [r.content for r in references]
                prompt_references = PROMPT_KNOWLEDGE.format(references="\n\n".join(references)).strip()
                system_prompt_parts.append(prompt_references)

        if context is not None:
            prompt_context = PROMPT_CONTEXT.format(context=self._format_context(context)).strip()
            system_prompt_parts.append(prompt_context)

        return "\n\n".join(system_prompt_parts).strip()

    def _reasoning_instructions(self) -> List[str]:
        """Instructions for the planning sub-agent used by `reason()`.

        The planner gets NO executable tools: with a team, calling a handoff
        during planning would trigger a real delegation (side effects) before
        the plan even exists. It receives the tool list as text instead, so
        the plan can still reference the capabilities available.
        """
        instructions = [
            "You are a meticulous and thoughtful assistant that solves a problem by thinking through it step-by-step.",
            "Carefully analyze the task by spelling it out loud.",
            "Then break down the problem by thinking through it step by step and develop multiple strategies to solve the problem.",
            "Your task is to provide a step-by-step plan, not to solve the problem yourself.",
        ]
        tools = self.get_all_tools()
        if tools:
            described = []
            for tool in tools:
                first_line = (tool.description or "").strip().split("\n")[0]
                described.append(f"{tool.name}: {first_line}" if first_line else tool.name)
            instructions.append(
                "The agent executing your plan has access to the following tools; reference them in the plan "
                "but do not try to call them yourself: " + "; ".join(described)
            )
        return instructions

    def reason(
        self,
        input: Union[str, List[Message], List[Dict[str, str]]],
    ) -> AgentResponse:
        if self.reasoning_model is None:
            return None

        logger.debug("Reasoning...")

        reasoning_agent = Agent(
            name="ReasoningAgent",
            instructions=self._reasoning_instructions(),
            llm=self.reasoning_model,
        )

        response = reasoning_agent.run(input)

        plan = (response.content or "").strip()
        if not plan:
            return None

        self.history.add_message(
            Message(
                role="assistant",
                content=f"I have worked through this problem in-depth and my reasoning is summarized below.\n\n{plan}",
            )
        )

        logger.debug("Thought:\n" + plan)

        return response

    async def areason(
        self,
        input: Union[str, List[Message], List[Dict[str, str]]],
    ) -> AgentResponse:
        if self.reasoning_model is None:
            return None

        logger.debug("Reasoning...")

        reasoning_agent = Agent(
            name="ReasoningAgent",
            instructions=self._reasoning_instructions(),
            llm=self.reasoning_model,
        )

        response = await reasoning_agent.arun(input)

        plan = (response.content or "").strip()
        if not plan:
            return None

        self.history.add_message(
            Message(
                role="assistant",
                content=f"I have worked through this problem in-depth and my reasoning is summarized below.\n\n{plan}",
            )
        )

        logger.debug("Thought:\n" + plan)

        return response

    def get_all_tools(self) -> list[BaseTool]:
        """All agent tools, including MCP tools and function tools."""
        tools = list(self.tools)
        for server in self.mcp_servers:
            tools += server.get_tools()
        return tools

    def _calibrate_history(self, window: List[Message], usage: Any) -> None:
        """Anchor the memory buffer on the provider's reported usage.

        Called once the reply is stored, so the anchor covers the request and
        its reply (see `ChatMemoryBuffer.observe_usage`).
        """
        observe = getattr(self.history, "observe_usage", None)
        input_tokens = getattr(usage, "input_tokens", 0) if usage is not None else 0
        output_tokens = getattr(usage, "output_tokens", 0) if usage is not None else 0
        if observe is not None and input_tokens:
            observe(window, input_tokens, output_tokens)

    def _tool_output_reader(self) -> Tool:
        store = self._tool_outputs
        default_length = (self.max_tool_output_tokens or 1000) * 4

        def read_tool_output(ref: str, offset: int = 0, length: int = default_length) -> str:
            """Read part of a long tool result that was set aside.

            Args:
                ref: The reference given in the truncated result.
                offset: Character offset to start reading from.
                length: Number of characters to read.
            """
            text = store.get(ref)
            if text is None:
                return f"Unknown reference '{ref}'."
            offset = max(int(offset or 0), 0)
            length = min(max(int(length or default_length), 1), default_length)
            chunk = text[offset : offset + length]
            end = offset + len(chunk)
            if end < len(text):
                chunk += f"\n[{end} of {len(text)} characters read; continue with offset={end}]"
            else:
                chunk += f"\n[end of output, {len(text)} characters]"
            return chunk

        return Tool(read_tool_output)

    def _set_aside_long_output(self, tool_name: str, content: str) -> str:
        """Apply `max_tool_output_tokens` to one tool result."""
        limit = self.max_tool_output_tokens
        if not limit or tool_name == READ_TOOL_OUTPUT or not content:
            return content
        tokenizer_fn = self.history.tokenizer_fn
        total = tokenizer_fn(content)
        if total <= limit:
            return content
        ref = f"out_{new_id().replace('-', '')[-8:]}"  # random tail: a uuid7 starts with the clock
        self._tool_outputs[ref] = content
        while len(self._tool_outputs) > MAX_STORED_TOOL_OUTPUTS:
            self._tool_outputs.pop(next(iter(self._tool_outputs)))
        head = keep_tokens_from_text(content, token_limit=limit, tokenizer_fn=tokenizer_fn)
        return (
            f"{head}\n[Output truncated: {total} tokens in total, {len(content)} characters. "
            f"Read the rest with {READ_TOOL_OUTPUT}(ref='{ref}', offset={len(head)}).]"
        )

    @tracer.tool(name="Tool Call")
    def run_tool(self, tool_call: ToolCall) -> Message:
        tool_execution = ToolResponse(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
        )

        tool_map = {tool.name: tool for tool in self.get_all_tools()}

        if tool_call.name == FINAL_RESULT_TOOL and self.response_model is not None:
            _, _, error_msg = self._validate_final_response(self._parse_tool_args(tool_call))
            logger.warning(f"final_result arguments failed validation: {error_msg}")
            tool_execution.content = error_msg or "Invalid final_result arguments"
            tracer.mark_error(message=tool_execution.content, error_type="invalid_final_result")
            return tool_execution.to_message()

        if tool_call.name not in tool_map.keys():
            logger.error(f"Tool {tool_call.name} does not exist")
            tracer.mark_error(message=f"Tool {tool_call.name} does not exist", error_type="unknown_tool")
            tool_execution.content = f"Tool {tool_call.name} does not exist"
            if self.response_model is not None:
                tool_execution.content += f". To give your final answer, call the `{FINAL_RESULT_TOOL}` tool."
            return tool_execution.to_message()

        if self.skills and tool_call.name == "load_skill":
            try:
                args = self._parse_tool_args(tool_call)
                skill_name = args.get("skill_name")
                logger.debug(f"[SKILL CALL] Loading skill '{skill_name}'")
                skill = next((s for s in self.skills if s.name == skill_name), None)
                if skill:
                    tool_execution.content = skill.to_prompt()
                    return tool_execution.to_message()
            except Exception as e:
                logger.error(f"Error loading skill '{tool_call.name}': {e}")

        try:
            tool = tool_map[tool_call.name]
            arguments = tool_call.arguments
            if not isinstance(arguments, dict):
                arguments = json.loads(arguments) if arguments and arguments.strip() else {}
            logger.info(f"[Tool Call] '{tool_call.name}'({arguments})")
            token = _CURRENT_TOOL_CALL_ID.set(tool_call.id)
            try:
                result = tool.run(**arguments)
            finally:
                _CURRENT_TOOL_CALL_ID.reset(token)
        except _SubagentPausedError:
            return None
        except Exception as e:
            logger.error(f"Error executing tool '{tool_call.name}': {e}")
            tracer.mark_error(e)
            tool_execution.content = f"Error executing tool '{tool_call.name}': {e}"
            return tool_execution.to_message()

        if result is None:
            tool_execution.content = ""
        elif isinstance(result, str):
            tool_execution.content = result
        else:
            tool_execution.content = str(result)
        tool_execution.content = self._set_aside_long_output(tool_call.name, tool_execution.content)
        return tool_execution.to_message()

    async def arun_tool(self, tool_call: ToolCall) -> Message:
        """Async twin of `run_tool` for tools whose `arun` does the work (durable proxies)."""
        tool_execution = ToolResponse(tool_call_id=tool_call.id, tool_name=tool_call.name)
        tool_map = {tool.name: tool for tool in self.get_all_tools()}
        if tool_call.name == FINAL_RESULT_TOOL and self.response_model is not None:
            return self.run_tool(tool_call)  # pure validation, no I/O
        if tool_call.name not in tool_map:
            return self.run_tool(tool_call)  # error message, no I/O
        try:
            result = await tool_map[tool_call.name].arun(**self._parse_tool_args(tool_call))
        except Exception as e:
            if in_workflow() and not isinstance(e, ActivityError):
                raise  # a bug in workflow code must fail the task, not be shown to the model
            logger.error(f"Error executing tool '{tool_call.name}': {e}")
            tool_execution.content = f"Error executing tool '{tool_call.name}': {getattr(e, 'cause', None) or e}"
            return tool_execution.to_message()
        tool_execution.content = self._set_aside_long_output(tool_call.name, result or "")
        return tool_execution.to_message()

    def execute_tool_calls(self, tool_calls: List[ToolCall]) -> List[Message]:
        delegation_names = {t.name for t in self._handoff_tools + self._spawn_tools}
        concurrent = [i for i, tool_call in enumerate(tool_calls) if tool_call.name in delegation_names]
        if len(concurrent) < 2:
            return [result for result in (self.run_tool(tool_call) for tool_call in tool_calls) if result]

        with ThreadPoolExecutor(max_workers=len(concurrent)) as pool:
            futures: Dict[int, Future] = {
                i: pool.submit(contextvars.copy_context().run, self.run_tool, tool_calls[i]) for i in concurrent
            }
            results = []
            for i, tool_call in enumerate(tool_calls):
                result = futures[i].result() if i in futures else self.run_tool(tool_call)
                if result:
                    results.append(result)
        return results

    async def aexecute_tool_calls(self, tool_calls: List[ToolCall]) -> List[Message]:
        if in_workflow():
            # No threads in workflow code: the calls are awaited, each one an activity.
            results = await asyncio.gather(*(self.arun_tool(tool_call) for tool_call in tool_calls))
            return [result for result in results if result]
        tasks = []
        for tool_call in tool_calls:
            task = asyncio.create_task(asyncio.to_thread(self.run_tool, tool_call))
            tasks.append(task)

        results = await asyncio.gather(*tasks)

        return [result for result in results if result]

    def _pending_tool_calls(self) -> List[ToolCall]:
        """Tool calls of the last assistant turn that have no result yet.

        Read from the history alone: a run paused on approvals keeps no other
        state, so it can be resumed from a restored history or in another process.
        """
        answered: set[str] = set()
        for message in reversed(self.history.messages):
            if message.role == "tool":
                answered.add(message.tool_call_id)
                continue
            if message.role != "assistant" or not message.tool_calls:
                return []
            return [
                ToolCall(id=call["id"], name=call["function"]["name"], arguments=call["function"]["arguments"])
                for call in message.tool_calls
                if call["id"] not in answered
            ]
        return []

    def _approval_request(self, tool_call: ToolCall) -> ToolApprovalRequest:
        return ToolApprovalRequest(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            tool_args=self._parse_tool_args(tool_call),
        )

    def get_pending_approvals(self) -> List[ToolApprovalRequest]:
        """The tool calls waiting for the user's approval, e.g. to ask again after reloading a session."""
        requests: List[ToolApprovalRequest] = []
        for tool_call in self._pending_tool_calls():
            paused = self._paused_subagents.get(tool_call.id)
            requests.extend(paused.requests if paused else [self._approval_request(tool_call)])
        return requests

    def _split_tool_calls(self, tool_calls: List[ToolCall]) -> tuple[List[ToolCall], List[ToolCall]]:
        """Separate the tool calls to run now from those that wait for the user's approval."""
        gated = {tool.name for tool in self.get_all_tools() if tool.requires_approval}
        return [t for t in tool_calls if t.name not in gated], [t for t in tool_calls if t.name in gated]

    def _decide_pending(
        self, approvals: Optional[ToolApprovals], resuming: bool = True
    ) -> tuple[List[ToolCall], List[Message]]:
        """Apply the user's decisions to the pending tool calls.

        Returns the approved calls, to execute, and the tool results of the
        denied ones. A pending call left without a decision is denied too: the
        user moved on without answering, and the providers reject a history
        where a tool call has no result.
        """
        approvals = approvals or {}
        pending = self._pending_tool_calls()
        pending_ids = {tool_call.id for tool_call in pending}
        for call_id in [key for key in self._paused_subagents if key not in pending_ids]:
            self._paused_subagents.pop(call_id)
        subagent_ids = {r.tool_call_id for paused in self._paused_subagents.values() for r in paused.requests}
        self._subagent_approvals = {key: value for key, value in approvals.items() if key in subagent_ids}
        unknown = set(approvals) - pending_ids - subagent_ids
        if unknown:
            logger.warning(f"Ignoring approvals for tool calls that are not pending: {sorted(unknown)}")

        approved: List[ToolCall] = []
        denials: List[Message] = []
        for tool_call in pending:
            if tool_call.id in self._paused_subagents:
                if resuming:
                    approved.append(tool_call)
                    continue
                self._paused_subagents.pop(tool_call.id)
                denials.append(
                    ToolResponse(
                        tool_call_id=tool_call.id, tool_name=tool_call.name, content=SUBAGENT_ABANDONED
                    ).to_message()
                )
                continue
            decision = approvals.get(tool_call.id)
            if decision is True:
                approved.append(tool_call)
                continue
            if isinstance(decision, ToolDenied):
                reason = decision.message
            elif decision is False:
                reason = ToolDenied().message
            else:
                reason = UNANSWERED_APPROVAL
            logger.info(f"[Tool Call] '{tool_call.name}' denied: {reason}")
            denials.append(
                ToolResponse(tool_call_id=tool_call.id, tool_name=tool_call.name, content=reason).to_message()
            )
        return approved, denials

    def _start_tool_calls(
        self, tool_calls: List[ToolCall], agent_response: AgentResponse
    ) -> Iterator[AgentResponseEvent]:
        for tool_call in tool_calls:
            event = AgentEventToolStarted(
                agent_id=self.id,
                run_id=agent_response.run_id,
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                tool_args=self._parse_tool_args(tool_call),
            )
            agent_response.events.append(event)
            yield event

    def _record_tool_results(
        self, tool_messages: List[Message], agent_response: AgentResponse
    ) -> Iterator[AgentResponseEvent]:
        for m in tool_messages:
            self.history.add_message(m)
            agent_response.messages.append(m)
            event = AgentEventToolCompleted(
                agent_id=self.id, run_id=agent_response.run_id, tool_call_id=m.tool_call_id, content=m.content
            )
            agent_response.events.append(event)
            yield event

    def _run_tool_round(
        self, tool_calls: List[ToolCall], agent_response: AgentResponse
    ) -> Iterator[AgentResponseEvent]:
        """Execute `tool_calls` and record their results in the history and the response.

        A generator so the streaming paths can yield each event as it happens;
        the other paths just exhaust it.
        """
        if not tool_calls:
            return
        yield from self._start_tool_calls(tool_calls, agent_response)
        tool_messages = yield from self._execute_with_subagent_events(tool_calls, agent_response)
        agent_response.usage.tool_calls += len(tool_messages)
        yield from self._record_tool_results(tool_messages, agent_response)
        if self.team or self.subagents:
            # Fold the sub-agents' spend into this run's usage.
            self._fold_delegation_usage(agent_response)
        yield from self._request_subagent_approvals(tool_calls, agent_response)

    def _execute_with_subagent_events(self, tool_calls: List[ToolCall], agent_response: AgentResponse):
        live = self._stream_subagent_events and any(t.name == SPAWN_AGENT_TOOL for t in tool_calls)
        if not live:
            tool_messages = self.execute_tool_calls(tool_calls=tool_calls)
            yield from self._drain_subagent_events(agent_response)
            return tool_messages
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            future = pool.submit(contextvars.copy_context().run, self.execute_tool_calls, tool_calls)
            while not future.done():
                try:
                    event = self._subagent_events.get(timeout=SUBAGENT_EVENT_POLL_SECONDS)
                except queue.Empty:
                    continue
                agent_response.events.append(event)
                yield event
            tool_messages = future.result()
        finally:
            pool.shutdown(wait=False)
        yield from self._drain_subagent_events(agent_response)
        return tool_messages

    def _drain_subagent_events(self, agent_response: AgentResponse) -> Iterator[AgentResponseEvent]:
        if self._subagent_events is None:
            return
        while True:
            try:
                event = self._subagent_events.get_nowait()
            except queue.Empty:
                return
            agent_response.events.append(event)
            yield event

    def _request_subagent_approvals(
        self, tool_calls: List[ToolCall], agent_response: AgentResponse
    ) -> Iterator[AgentResponseEvent]:
        for tool_call in tool_calls:
            paused = self._paused_subagents.get(tool_call.id)
            if paused is None:
                continue
            for request in paused.requests:
                agent_response.pending_approvals.append(request)
                event = AgentEventToolApprovalRequired(
                    agent_id=self.id,
                    run_id=agent_response.run_id,
                    tool_call_id=request.tool_call_id,
                    tool_name=request.tool_name,
                    tool_args=request.tool_args,
                    metadata={"subagent_call_id": tool_call.id},
                )
                agent_response.events.append(event)
                yield event

    def _open_subagent_events(self, agent_response: AgentResponse, stream: bool) -> None:
        self._subagent_events = queue.SimpleQueue() if self.subagents else None
        self._stream_subagent_events = stream and bool(self.subagents)
        self._subagent_run_id = agent_response.run_id

    def _can_continue(self, agent_response: AgentResponse) -> bool:
        return not agent_response.pending_approvals and not self._stop_requested.is_set()

    def _end_reason(self, completed: bool, paused: bool) -> str:
        if completed:
            return "stop"
        if paused:
            return "approval_required"
        return "interrupted" if self._stop_requested.is_set() else "max_turns"

    async def _arun_tool_round(
        self, tool_calls: List[ToolCall], agent_response: AgentResponse
    ) -> AsyncIterator[AgentResponseEvent]:
        """Async twin of `_run_tool_round`: the calls of the round run in parallel."""
        if not tool_calls:
            return
        for event in self._start_tool_calls(tool_calls, agent_response):
            yield event
        task = asyncio.ensure_future(self.aexecute_tool_calls(tool_calls=tool_calls))
        try:
            while not task.done():
                await asyncio.wait({task}, timeout=SUBAGENT_EVENT_POLL_SECONDS)
                for event in self._drain_subagent_events(agent_response):
                    yield event
        except BaseException:
            task.cancel()
            raise
        tool_messages = task.result()
        for event in self._drain_subagent_events(agent_response):
            yield event
        agent_response.usage.tool_calls += len(tool_messages)
        for event in self._record_tool_results(tool_messages, agent_response):
            yield event
        if self.team or self.subagents:
            self._fold_delegation_usage(agent_response)
        for event in self._request_subagent_approvals(tool_calls, agent_response):
            yield event

    def _request_approvals(
        self, tool_calls: List[ToolCall], agent_response: AgentResponse
    ) -> Iterator[AgentResponseEvent]:
        """Register the tool calls the run pauses on, one event per call."""
        for tool_call in tool_calls:
            request = self._approval_request(tool_call)
            agent_response.pending_approvals.append(request)
            event = AgentEventToolApprovalRequired(
                agent_id=self.id,
                run_id=agent_response.run_id,
                tool_call_id=request.tool_call_id,
                tool_name=request.tool_name,
                tool_args=request.tool_args,
            )
            agent_response.events.append(event)
            yield event

    def _resume_tool_calls(
        self, approvals: Optional[ToolApprovals], agent_response: AgentResponse, resuming: bool = True
    ) -> Iterator[AgentResponseEvent]:
        """Settle the tool calls a previous run paused on, before new input reaches the history."""
        approved, denials = self._decide_pending(approvals, resuming)
        yield from self._run_tool_round(approved, agent_response)
        yield from self._record_tool_results(denials, agent_response)

    async def _aresume_tool_calls(
        self, approvals: Optional[ToolApprovals], agent_response: AgentResponse, resuming: bool = True
    ) -> AsyncIterator[AgentResponseEvent]:
        approved, denials = self._decide_pending(approvals, resuming)
        async for event in self._arun_tool_round(approved, agent_response):
            yield event
        for event in self._record_tool_results(denials, agent_response):
            yield event

    def _run_input(self, input: Union[str, List[Message], List[Dict[str, str]], None]) -> tuple[List[Message], Any]:
        """The new messages of a run and its task.

        Without `input`, the run resumes the tool calls waiting for approval
        and the task (used for retrieval) is the last user message.
        """
        if input is not None:
            messages = self.llm.input_to_message_list(input)
            return messages, messages[-1].content
        if not self._pending_tool_calls():
            raise ValueError("`input` is required: no tool call is waiting for approval.")
        task = next((m.content for m in reversed(self.history.messages) if m.role == "user"), None)
        return [], task

    @tracer.agent(name="Agent Run")
    def run(
        self,
        input: Union[str, List[Message], List[Dict[str, str]], None] = None,
        context: Optional[Union[str, Dict[str, str]]] = None,
        approvals: Optional[ToolApprovals] = None,
    ) -> AgentResponse:
        if in_workflow():
            raise RuntimeError(
                "Inside a workflow, use `await agent.arun(...)`: synchronous and streaming runs are not durable."
            )
        messages, task = self._run_input(input)

        # init agent response
        agent_response = AgentResponse(agent_id=self.id)
        agent_response.events.append(
            AgentEventStarted(
                agent_id=self.id,
                run_id=agent_response.run_id,
            )
        )

        # history
        self._prepare_history(task=task, context=context)
        self._open_subagent_events(agent_response, stream=False)
        if messages:
            self._pin_task(messages)
        for _ in self._resume_tool_calls(approvals, agent_response, resuming=not messages):
            pass
        self.history.add_messages(messages)

        if self.reasoning_model and messages:
            self.history.compact()
            messages_for_reasoning_model = [m.to_dict() for m in self.history.get()]
            response = self.reason(messages_for_reasoning_model)
            if response is not None:
                agent_response.reasoning_content = response.reasoning_content
                agent_response.usage.add(response.usage)
                if response.reasoning_content:
                    agent_response.events.append(
                        AgentEventThinking(
                            agent_id=self.id,
                            run_id=agent_response.run_id,
                            content=response.reasoning_content,
                        )
                    )

        num_turns_available = self.max_turns
        validation_retries_left = MAX_VALIDATION_RETRIES
        force_structured = False
        completed = False
        response = None

        while num_turns_available > 0 and self._can_continue(agent_response):
            num_turns_available -= 1
            self._configure_llm_for_turn(structured=force_structured)
            self._before_request(num_turns_available)

            # format messages
            self.history.compact()
            window = self.history.get()
            messages_for_model = [m.to_dict() for m in window]

            # call llm and tool
            response = self.llm.invoke(input=messages_for_model)
            self._absorb_answer_tool_call(response)

            # usage
            agent_response.usage.add(response.usage)

            # native thinking from the main LLM call
            if response.thinking:
                if agent_response.reasoning_content:
                    agent_response.reasoning_content += "\n\n" + response.thinking
                else:
                    agent_response.reasoning_content = response.thinking
                agent_response.events.append(
                    AgentEventThinking(
                        agent_id=self.id,
                        run_id=agent_response.run_id,
                        content=response.thinking,
                    )
                )

            if response.content:
                agent_response.events.append(
                    AgentEventContent(
                        agent_id=self.id,
                        run_id=agent_response.run_id,
                        content=response.content,
                    )
                )

            # keep answer in memory (with provider-shaped thinking blocks for echo-back)
            tool_calls = [t.to_message_dict() for t in response.tool_calls]
            _message = Message(role="assistant", content=response.content, tool_calls=tool_calls)
            _message.thinking_parts = self.llm.get_thinking_parts(response)
            self.history.add_message(_message)
            self._calibrate_history(window, response.usage)

            # stop if not tool call
            if not response.tool_calls:
                content_to_check = response.parsed if response.parsed else response.content
                is_valid, parsed_data, error_msg = self._validate_final_response(content_to_check)

                if not is_valid and validation_retries_left > 0:
                    validation_retries_left -= 1
                    force_structured = True
                    retry_message = Message(role="user", content=error_msg)
                    self.history.add_message(retry_message)
                    continue

                if not is_valid:
                    logger.error(
                        f"Final response still fails schema validation after "
                        f"{MAX_VALIDATION_RETRIES} retries: {error_msg}"
                    )
                agent_response.content = response.content
                agent_response.parsed = parsed_data
                agent_response.messages.append(_message)
                completed = True
                break

            if response.tool_calls and self.get_all_tools():
                to_run, to_approve = self._split_tool_calls(response.tool_calls)
                for _ in self._run_tool_round(to_run, agent_response):
                    pass
                for _ in self._request_approvals(to_approve, agent_response):
                    pass

            if self.tool_choice == "required":
                self.tool_choice = "auto"
                self.llm.tool_choice = "auto"

            if agent_response.pending_approvals:
                break

        paused = bool(agent_response.pending_approvals)
        if paused:
            agent_response.content = (response.content or None) if response is not None else None
        elif not completed:
            agent_response.content = None
            logger.warning(f"Agent stopped after max_turns={self.max_turns} without a final answer.")

        agent_response.events.append(
            AgentEventCompleted(
                agent_id=self.id,
                run_id=agent_response.run_id,
            )
        )

        self._finish_run(agent_response, self._end_reason(completed, paused))

        return agent_response

    @tracer.agent(name="Agent Arun")
    async def arun(
        self,
        input: Union[str, List[Message], List[Dict[str, str]], None] = None,
        context: Optional[Union[str, Dict[str, str]]] = None,
        approvals: Optional[ToolApprovals] = None,
    ) -> AgentResponse:
        if in_workflow():
            if self.durability is None:
                raise RuntimeError(
                    f"Agent '{self.name}' runs inside a workflow without durability: "
                    "give it `durability=TemporalDurability()` so its I/O runs in activities."
                )
            if not getattr(self, "_workflow_copy", False):
                # Each workflow run gets its own copy (own history), rebuilt identically on replay.
                return await self.durability.workflow_agent(self).arun(input, context=context, approvals=approvals)
        messages, task = self._run_input(input)

        agent_response = AgentResponse(agent_id=self.id)
        agent_response.events.append(
            AgentEventStarted(
                agent_id=self.id,
                run_id=agent_response.run_id,
            )
        )

        self._prepare_history(task=task, context=context)
        self._open_subagent_events(agent_response, stream=False)
        if messages:
            self._pin_task(messages)
        async for _ in self._aresume_tool_calls(approvals, agent_response, resuming=not messages):
            pass
        self.history.add_messages(messages)

        if self.reasoning_model and messages:
            await self.history.acompact()
            messages_for_reasoning_model = [m.to_dict() for m in self.history.get()]
            reasoning_agent_response = await self.areason(messages_for_reasoning_model)
            if reasoning_agent_response is not None:
                agent_response.reasoning_content = reasoning_agent_response.reasoning_content
                agent_response.usage.add(reasoning_agent_response.usage)
                if reasoning_agent_response.reasoning_content:
                    agent_response.events.append(
                        AgentEventThinking(
                            agent_id=self.id,
                            run_id=agent_response.run_id,
                            content=reasoning_agent_response.reasoning_content,
                        )
                    )

        num_turns_available = self.max_turns
        validation_retries_left = MAX_VALIDATION_RETRIES
        force_structured = False
        completed = False
        response = None

        while num_turns_available > 0 and self._can_continue(agent_response):
            num_turns_available -= 1
            self._configure_llm_for_turn(structured=force_structured)
            self._before_request(num_turns_available)

            await self.history.acompact()
            window = self.history.get()
            messages_for_model = [m.to_dict() for m in window]

            response = await self.llm.ainvoke(input=messages_for_model)
            self._absorb_answer_tool_call(response)

            # usage
            agent_response.usage.add(response.usage)

            if response.thinking:
                if agent_response.reasoning_content:
                    agent_response.reasoning_content += "\n\n" + response.thinking
                else:
                    agent_response.reasoning_content = response.thinking
                agent_response.events.append(
                    AgentEventThinking(
                        agent_id=self.id,
                        run_id=agent_response.run_id,
                        content=response.thinking,
                    )
                )

            if response.content:
                agent_response.events.append(
                    AgentEventContent(
                        agent_id=self.id,
                        run_id=agent_response.run_id,
                        content=response.content,
                    )
                )

            tool_calls = [t.to_message_dict() for t in response.tool_calls]
            _message = Message(role="assistant", content=response.content, tool_calls=tool_calls)
            _message.thinking_parts = self.llm.get_thinking_parts(response)
            self.history.add_message(_message)
            self._calibrate_history(window, response.usage)

            if not response.tool_calls:
                content_to_check = response.parsed if response.parsed else response.content
                is_valid, parsed_data, error_msg = self._validate_final_response(content_to_check)

                if not is_valid and validation_retries_left > 0:
                    validation_retries_left -= 1
                    force_structured = True
                    retry_message = Message(role="user", content=error_msg)
                    self.history.add_message(retry_message)
                    continue

                if not is_valid:
                    logger.error(
                        f"Final response still fails schema validation after "
                        f"{MAX_VALIDATION_RETRIES} retries: {error_msg}"
                    )
                agent_response.content = response.content
                agent_response.parsed = parsed_data
                agent_response.messages.append(_message)
                completed = True
                break

            if response.tool_calls and self.get_all_tools():
                to_run, to_approve = self._split_tool_calls(response.tool_calls)
                async for _ in self._arun_tool_round(to_run, agent_response):
                    pass
                for _ in self._request_approvals(to_approve, agent_response):
                    pass

            if self.tool_choice == "required":
                self.tool_choice = "auto"
                self.llm.tool_choice = "auto"

            if agent_response.pending_approvals:
                break

        paused = bool(agent_response.pending_approvals)
        if paused:
            agent_response.content = (response.content or None) if response is not None else None
        elif not completed:
            agent_response.content = None
            logger.warning(f"Agent stopped after max_turns={self.max_turns} without a final answer.")

        agent_response.events.append(
            AgentEventCompleted(
                agent_id=self.id,
                run_id=agent_response.run_id,
            )
        )

        self._finish_run(agent_response, self._end_reason(completed, paused))

        return agent_response

    @tracer.agent(name="Agent Stream")
    def run_stream(
        self,
        input: Union[str, List[Message], List[Dict[str, str]], None] = None,
        context: Optional[Union[str, Dict[str, str]]] = None,
        approvals: Optional[ToolApprovals] = None,
    ) -> Iterator[AgentResponse]:
        if in_workflow():
            raise RuntimeError(
                "Inside a workflow, use `await agent.arun(...)`: synchronous and streaming runs are not durable."
            )
        agent_response = AgentResponse(agent_id=self.id)
        events = self._run_stream(agent_response, input, context, approvals)
        try:
            yield from events
        except GeneratorExit:
            self._interrupt_run(agent_response)
            raise
        finally:
            events.close()

    def _run_stream(
        self,
        agent_response: AgentResponse,
        input: Union[str, List[Message], List[Dict[str, str]], None],
        context: Optional[Union[str, Dict[str, str]]],
        approvals: Optional[ToolApprovals],
    ) -> Iterator[AgentResponse]:
        messages, task = self._run_input(input)

        event = AgentEventStarted(
            agent_id=self.id,
            run_id=agent_response.run_id,
        )
        agent_response.events.append(event)
        yield event

        self._prepare_history(task=task, context=context)
        self._open_subagent_events(agent_response, stream=True)
        if messages:
            self._pin_task(messages)
        yield from self._resume_tool_calls(approvals, agent_response, resuming=not messages)
        self.history.add_messages(messages)

        if self.reasoning_model and messages:
            self.history.compact()
            messages_for_reasoning_model = [m.to_dict() for m in self.history.get()]
            reasoning_response = self.reason(messages_for_reasoning_model)
            if reasoning_response is not None and reasoning_response.reasoning_content:
                agent_response.reasoning_content = reasoning_response.reasoning_content
                agent_response.usage.add(reasoning_response.usage)
                event = AgentEventThinking(
                    agent_id=self.id, run_id=agent_response.run_id, content=reasoning_response.reasoning_content
                )
                agent_response.events.append(event)
                yield event

        num_turns_available = self.max_turns
        validation_retries_left = MAX_VALIDATION_RETRIES
        force_structured = False
        completed = False
        full_content = ""

        while num_turns_available > 0 and self._can_continue(agent_response):
            num_turns_available -= 1
            self._configure_llm_for_turn(structured=force_structured)
            self._before_request(num_turns_available)

            self.history.compact()
            window = self.history.get()
            messages_for_model = [m.to_dict() for m in window]

            full_content = ""
            full_thinking = ""
            final_tool_calls: List[ToolCall] = []
            turn_usage = RequestUsage()
            turn_response = ModelResponse()

            for chunk in self.llm.stream(input=messages_for_model):
                if chunk.usage:
                    turn_usage += chunk.usage

                agent_response.content = None

                if chunk.thinking:
                    full_thinking += chunk.thinking
                    event = AgentEventThinking(
                        agent_id=self.id,
                        run_id=agent_response.run_id,
                        content=chunk.thinking,
                    )
                    agent_response.events.append(event)
                    yield event

                if chunk.content:
                    agent_response.content = chunk.content
                    full_content += chunk.content
                    event = AgentEventContent(agent_id=self.id, run_id=agent_response.run_id, content=chunk.content)
                    agent_response.events.append(event)
                    yield event

                if chunk.tool_calls:
                    final_tool_calls = chunk.tool_calls

                turn_response.parts.extend(chunk.parts)

            agent_response.usage.add(turn_usage)

            answer = self._answer_from_tool_calls(final_tool_calls)
            if answer is not None:
                full_content, final_tool_calls = answer, []
                event = AgentEventContent(agent_id=self.id, run_id=agent_response.run_id, content=answer)
                agent_response.events.append(event)
                yield event

            if full_thinking:
                if agent_response.reasoning_content:
                    agent_response.reasoning_content += "\n\n" + full_thinking
                else:
                    agent_response.reasoning_content = full_thinking

            tool_calls_dicts = [t.to_message_dict() for t in final_tool_calls]
            _message = Message(role="assistant", content=full_content, tool_calls=tool_calls_dicts)
            _message.thinking_parts = self.llm.get_thinking_parts(turn_response)
            self.history.add_message(_message)
            self._calibrate_history(window, turn_usage)

            if not final_tool_calls:
                if self.response_model:
                    is_valid, parsed_data, error_msg = self._validate_final_response(full_content)
                    if not is_valid and validation_retries_left > 0:
                        validation_retries_left -= 1
                        force_structured = True
                        self.history.add_message(Message(role="user", content=error_msg))
                        continue
                    if not is_valid:
                        logger.error(
                            f"Final response still fails schema validation after "
                            f"{MAX_VALIDATION_RETRIES} retries: {error_msg}"
                        )
                    agent_response.parsed = parsed_data
                agent_response.content = full_content
                agent_response.messages.append(_message)
                completed = True
                break

            if final_tool_calls and self.get_all_tools():
                to_run, to_approve = self._split_tool_calls(final_tool_calls)
                yield from self._run_tool_round(to_run, agent_response)
                yield from self._request_approvals(to_approve, agent_response)

            if self.tool_choice == "required":
                self.tool_choice = "auto"
                self.llm.tool_choice = "auto"

            if agent_response.pending_approvals:
                break

        paused = bool(agent_response.pending_approvals)
        if paused:
            agent_response.content = full_content or None
        elif not completed:
            agent_response.content = None
            logger.warning(f"Agent stopped after max_turns={self.max_turns} without a final answer.")

        self._finish_run(agent_response, self._end_reason(completed, paused))
        tracer.record_agent_result(agent_response)
        event = AgentEventCompleted(
            agent_id=self.id,
            run_id=agent_response.run_id,
        )
        agent_response.events.append(event)
        yield event

        yield agent_response

    @tracer.agent(name="Agent Astream")
    async def arun_stream(
        self,
        input: Union[str, List[Message], List[Dict[str, str]], None] = None,
        context: Optional[Union[str, Dict[str, str]]] = None,
        approvals: Optional[ToolApprovals] = None,
    ) -> AsyncIterator[AgentResponse]:
        if in_workflow():
            raise RuntimeError(
                "Inside a workflow, use `await agent.arun(...)`: synchronous and streaming runs are not durable."
            )
        agent_response = AgentResponse(agent_id=self.id)
        events = self._arun_stream(agent_response, input, context, approvals)
        try:
            async for event in events:
                yield event
        except (GeneratorExit, asyncio.CancelledError):
            self._interrupt_run(agent_response)
            raise
        finally:
            await events.aclose()

    async def _arun_stream(
        self,
        agent_response: AgentResponse,
        input: Union[str, List[Message], List[Dict[str, str]], None],
        context: Optional[Union[str, Dict[str, str]]],
        approvals: Optional[ToolApprovals],
    ) -> AsyncIterator[AgentResponse]:
        messages, task = self._run_input(input)

        event = AgentEventStarted(
            agent_id=self.id,
            run_id=agent_response.run_id,
        )
        agent_response.events.append(event)
        yield event

        self._prepare_history(task=task, context=context)
        self._open_subagent_events(agent_response, stream=True)
        if messages:
            self._pin_task(messages)
        async for event in self._aresume_tool_calls(approvals, agent_response, resuming=not messages):
            yield event
        self.history.add_messages(messages)

        if self.reasoning_model and messages:
            await self.history.acompact()
            messages_for_reasoning_model = [m.to_dict() for m in self.history.get()]
            reasoning_response = await self.areason(messages_for_reasoning_model)
            if reasoning_response is not None and reasoning_response.reasoning_content:
                agent_response.reasoning_content = reasoning_response.reasoning_content
                agent_response.usage.add(reasoning_response.usage)
                event = AgentEventThinking(
                    agent_id=self.id, run_id=agent_response.run_id, content=reasoning_response.reasoning_content
                )
                agent_response.events.append(event)
                yield event

        num_turns_available = self.max_turns
        validation_retries_left = MAX_VALIDATION_RETRIES
        force_structured = False
        completed = False
        full_content = ""

        while num_turns_available > 0 and self._can_continue(agent_response):
            num_turns_available -= 1
            self._configure_llm_for_turn(structured=force_structured)
            self._before_request(num_turns_available)

            await self.history.acompact()
            window = self.history.get()
            messages_for_model = [m.to_dict() for m in window]

            full_content = ""
            full_thinking = ""
            final_tool_calls: List[ToolCall] = []
            turn_usage = RequestUsage()
            turn_response = ModelResponse()

            async for chunk in self.llm.astream(input=messages_for_model):
                if chunk.usage:
                    turn_usage += chunk.usage

                agent_response.content = None

                if chunk.thinking:
                    full_thinking += chunk.thinking
                    event = AgentEventThinking(
                        agent_id=self.id,
                        run_id=agent_response.run_id,
                        content=chunk.thinking,
                    )
                    agent_response.events.append(event)
                    yield event

                if chunk.content:
                    agent_response.content = chunk.content
                    full_content += chunk.content
                    event = AgentEventContent(agent_id=self.id, run_id=agent_response.run_id, content=chunk.content)
                    agent_response.events.append(event)
                    yield event

                if chunk.tool_calls:
                    final_tool_calls = chunk.tool_calls

                turn_response.parts.extend(chunk.parts)

            agent_response.usage.add(turn_usage)

            answer = self._answer_from_tool_calls(final_tool_calls)
            if answer is not None:
                full_content, final_tool_calls = answer, []
                event = AgentEventContent(agent_id=self.id, run_id=agent_response.run_id, content=answer)
                agent_response.events.append(event)
                yield event

            if full_thinking:
                if agent_response.reasoning_content:
                    agent_response.reasoning_content += "\n\n" + full_thinking
                else:
                    agent_response.reasoning_content = full_thinking

            tool_calls_dicts = [t.to_message_dict() for t in final_tool_calls]
            _message = Message(role="assistant", content=full_content, tool_calls=tool_calls_dicts)
            _message.thinking_parts = self.llm.get_thinking_parts(turn_response)
            self.history.add_message(_message)
            self._calibrate_history(window, turn_usage)

            if not final_tool_calls:
                if self.response_model:
                    is_valid, parsed_data, error_msg = self._validate_final_response(full_content)
                    if not is_valid and validation_retries_left > 0:
                        validation_retries_left -= 1
                        force_structured = True
                        self.history.add_message(Message(role="user", content=error_msg))
                        continue
                    if not is_valid:
                        logger.error(
                            f"Final response still fails schema validation after "
                            f"{MAX_VALIDATION_RETRIES} retries: {error_msg}"
                        )
                    agent_response.parsed = parsed_data
                agent_response.content = full_content
                agent_response.messages.append(_message)
                completed = True
                break

            if final_tool_calls and self.get_all_tools():
                to_run, to_approve = self._split_tool_calls(final_tool_calls)
                async for event in self._arun_tool_round(to_run, agent_response):
                    yield event
                for event in self._request_approvals(to_approve, agent_response):
                    yield event

            if self.tool_choice == "required":
                self.tool_choice = "auto"
                self.llm.tool_choice = "auto"

            if agent_response.pending_approvals:
                break

        paused = bool(agent_response.pending_approvals)
        if paused:
            agent_response.content = full_content or None
        elif not completed:
            agent_response.content = None
            logger.warning(f"Agent stopped after max_turns={self.max_turns} without a final answer.")

        self._finish_run(agent_response, self._end_reason(completed, paused))
        tracer.record_agent_result(agent_response)
        event = AgentEventCompleted(
            agent_id=self.id,
            run_id=agent_response.run_id,
        )
        agent_response.events.append(event)
        yield event

        yield agent_response
