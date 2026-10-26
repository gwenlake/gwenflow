import asyncio
import copy
import json
import math
import re
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
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

from gwenflow.agents.prompts import PROMPT_CONTEXT, PROMPT_JSON_SCHEMA, PROMPT_KNOWLEDGE
from gwenflow.llms import ChatBase, ChatOpenAI
from gwenflow.llms.base import CONTEXT_SAFETY_RATIO
from gwenflow.logger import logger
from gwenflow.memory import ChatMemoryBuffer, ChatSummaryMemoryBuffer, MemoryToolset
from gwenflow.memory.chat_memory_buffer import DEFAULT_MAX_FRACTION
from gwenflow.memory.compaction import PINNED_KEY, pin
from gwenflow.retriever import Retriever
from gwenflow.skills import Skill, SkillsToolset
from gwenflow.telemetry import tracer
from gwenflow.tools import BaseTool, MCPServer, Tool
from gwenflow.types import (
    AgentEventCompleted,
    AgentEventContent,
    AgentEventStarted,
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
from gwenflow.utils.tokens import keep_tokens_from_text

FINAL_RESULT_TOOL = "final_result"
READ_TOOL_OUTPUT = "read_tool_output"
MAX_STORED_TOOL_OUTPUTS = 50


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


AGENT_COMMANDS = {
    "/help": "Show the available commands.",
    "/compact": "Fold older messages into the running summary now.",
    "/reset": "Clear the conversation history and summary.",
    "/clear": "Alias of /reset.",
    "/history": "Show conversation memory statistics.",
    "/cost": "Show cumulative token usage and estimated cost.",
    "/tools": "List the registered tools.",
    "/summary": "Show the current running summary.",
    "/system": "Show the resolved system prompt.",
    "/config": "Show the agent configuration.",
    "/save": "Save the conversation (history + summary) to JSON: /save <path>.",
    "/load": "Load a conversation saved with /save: /load <path>.",
}

COMMANDS_WITH_ARGS = {"/save", "/load"}
"""Commands that accept an argument. Every other command is intercepted only
when the message is exactly the bare command, so that a real request that
merely starts with one ("/reset the counter in my code") still reaches the
LLM."""

SESSION_FORMAT_VERSION = 1
"""Version of the /save JSON format."""

MAX_VALIDATION_RETRIES = 3
"""How many times an invalid structured (`response_model`) answer is sent
back to the model for correction before giving up (parsed=None, raw
content kept)."""

ToolApprovals = Dict[str, Union[bool, ToolDenied]]
"""The user's decisions on the tool calls a run paused on, by tool call id:
True runs the call, False or a `ToolDenied` (with the reason the model reads)
refuses it. Only an explicit True approves."""

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

    max_turns: Optional[int] = 100
    """Maximum turn (tool calls, llm calls) an agent can do."""

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
    included (see /cost)."""

    pricing: Optional[Dict[str, float]] = None
    """Optional price grid in dollars per MILLION tokens, e.g.
    {"input": 3.0, "output": 15.0} (optional keys: "cache_read",
    "cache_write"). When set, /cost adds an estimated dollar amount."""

    delegation_usage: AgentUsage = field(default_factory=AgentUsage)
    """Cumulative usage of team handoffs (sub-agent runs). Informational
    breakdown for /cost: this spend is already folded into `total_usage`."""

    _pending_delegation_usage: AgentUsage = field(default_factory=AgentUsage, init=False, repr=False)
    """Delegation usage accumulated since the last drain. Guarded by
    `_delegation_lock` because handoffs may run in parallel threads
    (`aexecute_tool_calls`); the run loops drain it after each tool round."""

    _delegation_lock: Any = field(default_factory=threading.Lock, init=False, repr=False)

    _handoff_tools: List[BaseTool] = field(default_factory=list, init=False, repr=False)
    """The subset of `tools` that wraps team members, kept separate so
    `clone()` can rebuild them bound to the clone."""

    def __post_init__(self) -> None:
        if self.id is None:
            self.id = str(uuid.uuid4())
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
        if self.max_tool_output_tokens and not any(t.name == READ_TOOL_OUTPUT for t in self.tools):
            self.tools = list(self.tools) + [self._tool_output_reader()]
        if self.memory:
            self.tools = list(self.tools) + self.memory.get_tools()
        if self.team:
            self._handoff_tools = self._build_handoff_tools()
            self.tools = list(self.tools) + self._handoff_tools
        if self.tools or self.mcp_servers:
            self.llm.tools = self._llm_tools()
            self.llm.tool_choice = self.tool_choice
        else:
            self.llm.tools = None
            self.llm.tool_choice = None

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
                if response.finish_reason == "max_turns":
                    return (
                        f"[handoff failed] The '{member.name or 'agent'}' agent stopped "
                        f"after reaching its max_turns limit without a final answer."
                    )
                if response.finish_reason == "approval_required":
                    tools = ", ".join(r.tool_name for r in response.pending_approvals)
                    return (
                        f"[handoff failed] The '{member.name or 'agent'}' agent needs the user's "
                        f"approval to run {tools}, which a delegated task cannot ask for."
                    )
                return response.content or ""

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
        worker.id = str(uuid.uuid4())
        worker.llm = copy.copy(self.llm)
        worker.history = self._fresh_history()
        worker.total_usage = AgentUsage()
        worker.delegation_usage = AgentUsage()
        worker._pending_delegation_usage = AgentUsage()
        worker._delegation_lock = threading.Lock()
        if self.team:
            worker._handoff_tools = worker._build_handoff_tools()
            handoff_ids = {id(t) for t in self._handoff_tools}
            base_tools = [t for t in self.tools if id(t) not in handoff_ids]
            worker.tools = base_tools + worker._handoff_tools
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
        cumulative `delegation_usage` feeds the /cost breakdown.
        """
        usage = AgentUsage()
        usage.add(response.usage)  # the worker's compaction spend is already inside
        with self._delegation_lock:
            self._pending_delegation_usage.add(usage)
            self.delegation_usage.add(usage)

    def _drain_delegation_usage(self) -> AgentUsage:
        """Take and reset the delegation usage accumulated by handoffs."""
        with self._delegation_lock:
            drained = self._pending_delegation_usage
            self._pending_delegation_usage = AgentUsage()
        return drained

    # ------------------------------------------------------------------
    # Slash commands (/help, /compact, /reset, ...)
    # ------------------------------------------------------------------

    def _match_command(self, task: Any) -> tuple[str, str] | None:
        """Return `(command, args)` if `task` is a known slash command.

        A bare known command ("/reset", "  /COMPACT  ") is always intercepted.
        Extra text after the command is only allowed for COMMANDS_WITH_ARGS
        ("/save ./session.json"); for every other command a message that merely
        starts with one ("/reset the counter in my code") goes to the LLM like
        any other input. Commands are handled locally, without an LLM call, and
        are never added to the history.
        """
        if not isinstance(task, str):
            return None
        stripped = task.strip()
        if not stripped.startswith("/"):
            return None
        parts = stripped.split(maxsplit=1)
        command = parts[0].lower()
        args = parts[1].strip() if len(parts) > 1 else ""
        if command not in AGENT_COMMANDS:
            return None
        if args and command not in COMMANDS_WITH_ARGS:
            return None
        return command, args

    def _command_response(self, content: str) -> AgentResponse:
        agent_response = AgentResponse(agent_id=self.id)
        agent_response.events.append(AgentEventStarted(agent_id=self.id, run_id=agent_response.run_id))
        agent_response.events.append(AgentEventContent(agent_id=self.id, run_id=agent_response.run_id, content=content))
        agent_response.content = content
        agent_response.events.append(AgentEventCompleted(agent_id=self.id, run_id=agent_response.run_id))
        agent_response.finish_reason = "stop"
        return agent_response

    def _compact_report(self, record: Any) -> str:
        if record is None:
            return "Nothing to compact: the conversation fits in the context window."
        report = (
            f"Compacted the history ({', '.join(record.strategies)}): "
            f"{record.messages_before} -> {record.messages_after} message(s), "
            f"~{record.tokens_before:,} -> ~{record.tokens_after:,} tokens."
        )
        summary = getattr(self.history, "summary", "")
        if summary and "summarize_older" in record.strategies:
            report += f" Running summary: ~{self.history.tokenizer_fn(summary)} tokens."
        return report

    def _execute_command(self, command: str, args: str = "") -> str:
        if command == "/help":
            lines = ["Available commands:"]
            lines += [f"  {name:<10} {desc}" for name, desc in AGENT_COMMANDS.items()]
            return "\n".join(lines)
        if command in ("/reset", "/clear"):
            removed = len(self.history.messages)
            self.history.reset()
            return f"History cleared ({removed} message(s) removed)."
        if command == "/compact":
            record = self.history.compact()
            self.total_usage.add(self.history.drain_usage())
            return self._compact_report(record)
        if command == "/cost":
            return self._cost_report()
        if command == "/tools":
            tools = self.get_all_tools()
            if not tools:
                return "No tools registered."
            lines = [f"{len(tools)} tool(s) available:"]
            for tool in tools:
                first_line = (tool.description or "").strip().split("\n")[0]
                lines.append(f"  {tool.name}: {first_line}" if first_line else f"  {tool.name}")
            return "\n".join(lines)
        if command == "/summary":
            summary = getattr(self.history, "summary", None)
            if summary is None:
                return "This agent's memory does not keep a summary (plain ChatMemoryBuffer)."
            if not summary:
                return "The running summary is empty (nothing has been compacted yet)."
            return f"Running summary:\n\n{summary}"
        if command == "/system":
            if self.system_prompt:
                return f"System prompt (fixed):\n\n{self.system_prompt.strip()}"
            retriever, self.retriever = self.retriever, None
            try:
                prompt = self.get_system_prompt(task="")
            finally:
                self.retriever = retriever
            note = (
                "\n\n(Knowledge references from the retriever are injected per task and are not shown here.)"
                if retriever
                else ""
            )
            return (f"System prompt:\n\n{prompt}" if prompt else "System prompt is empty.") + note
        if command == "/config":
            return self._config_report()
        if command == "/save":
            if not args:
                return "Usage: /save <path>"
            return self._save_session(args)
        if command == "/load":
            if not args:
                return "Usage: /load <path>"
            return self._load_session(args)
        if command == "/history":
            messages = self.history.messages
            tokens = self.history._token_count_for_messages(messages)
            parts = [f"{len(messages)} message(s) stored, ~{tokens} tokens (limit {self.history.limit()})."]
            summary = getattr(self.history, "summary", "")
            if summary:
                parts.append(f"Running summary: ~{self.history.tokenizer_fn(summary)} tokens.")
            folded = getattr(self.history, "_summarized_upto", 0)
            if folded:
                parts.append(f"{folded} message(s) already folded into the summary.")
            return " ".join(parts)
        return f"Unknown command: {command}"

    def _config_report(self) -> str:
        lines = ["Agent configuration:"]
        lines.append(f"  name:            {self.name or '-'}")
        lines.append(f"  model:           {getattr(self.llm, 'model', type(self.llm).__name__)}")
        if self.reasoning_model is not None:
            lines.append(
                f"  reasoning model: {getattr(self.reasoning_model, 'model', type(self.reasoning_model).__name__)}"
            )
        lines.append(f"  max_turns:       {self.max_turns}")
        lines.append(f"  tool_choice:     {self.tool_choice or 'auto'}")
        lines.append(f"  tools:           {len(self.get_all_tools())}")
        if self.skills:
            lines.append(f"  skills:          {', '.join(s.name for s in self.skills)}")
        if self.memory:
            lines.append(
                f"  memory:          {len(self.memory.always)} core file(s), {len(self.memory.index())} topic file(s)"
            )
        if self.team:
            lines.append(f"  team:            {', '.join(m.name or m.id for m in self.team)}")
        if self.response_model is not None:
            model_name = self.response_model.__name__ if isinstance(self.response_model, type) else "dict schema"
            lines.append(f"  response_model:  {model_name}")
        lines.append(
            f"  memory:          {type(self.history).__name__} "
            f"(token_limit {self.history.limit():,}, reserved {self.history.reserved_tokens:,})"
        )
        summarizer = getattr(self.history, "llm", None)
        if summarizer is not None:
            lines.append(f"  summarizer:      {getattr(summarizer, 'model', type(summarizer).__name__)}")
        return "\n".join(lines)

    def _save_session(self, path: str) -> str:
        data: Dict[str, Any] = {
            "format": "gwenflow.session",
            "version": SESSION_FORMAT_VERSION,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "agent": {"id": self.id, "name": self.name},
            "messages": [m.to_dict() for m in self.history.messages],
        }
        summary = getattr(self.history, "summary", None)
        if summary is not None:
            data["summary"] = summary
            data["summarized_upto"] = self.history._summarized_upto
        try:
            file = Path(path).expanduser()
            if file.parent != Path("."):
                file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text(
                json.dumps(
                    data,
                    indent=2,
                    ensure_ascii=False,
                    default=lambda o: o.to_dict() if hasattr(o, "to_dict") else str(o),
                ),
                encoding="utf-8",
            )
        except OSError as e:
            return f"Could not save session: {e}"
        return f"Session saved to {file} ({len(self.history.messages)} message(s))."

    def _load_session(self, path: str) -> str:
        file = Path(path).expanduser()
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            return f"Could not load session: {e}"
        if not isinstance(data, dict) or data.get("format") != "gwenflow.session":
            return f"Could not load session: {file} is not a gwenflow session file."
        version = data.get("version", 0)
        if version > SESSION_FORMAT_VERSION:
            return f"Could not load session: file version {version} is newer than supported ({SESSION_FORMAT_VERSION})."

        messages = data.get("messages", [])
        self.history.reset()
        for message in messages:
            self.history.add_message(message)

        summary_note = ""
        if hasattr(self.history, "summary"):
            self.history.summary = data.get("summary") or ""
            self.history._summarized_upto = int(data.get("summarized_upto") or 0)
            if self.history.summary:
                summary_note = ", summary restored"
        elif data.get("summary"):
            summary_note = ", summary in file ignored (plain ChatMemoryBuffer)"
        return f"Session loaded from {file} ({len(messages)} message(s){summary_note})."

    def _cost_report(self) -> str:
        def block(title: str, usage: AgentUsage) -> list[str]:
            lines = [title]
            requests = f"  requests:      {usage.requests:,}"
            if usage.tool_calls:
                requests += f"  (tool calls: {usage.tool_calls:,})"
            lines.append(requests)
            input_line = f"  input tokens:  {usage.input_tokens:,}"
            if usage.cache_read_tokens:
                input_line += f"  (cache read: {usage.cache_read_tokens:,})"
            lines.append(input_line)
            lines.append(f"  output tokens: {usage.output_tokens:,}")
            return lines

        combined = AgentUsage()
        combined.add(self.total_usage)
        lines = block("Session usage:", self.total_usage)

        if self.delegation_usage.requests:
            # Breakdown only: this spend is already inside the session usage.
            lines += block("Team delegations (included above):", self.delegation_usage)

        compaction_usage = getattr(self.history, "usage", None)
        if compaction_usage is not None and compaction_usage.requests:
            # Breakdown only: summaries are folded into the run that triggered them.
            lines += block("Compaction (included above):", compaction_usage)

        if self.pricing:
            cost = combined.input_tokens * self.pricing.get("input", 0.0)
            cost += combined.output_tokens * self.pricing.get("output", 0.0)
            cost += combined.cache_read_tokens * self.pricing.get("cache_read", 0.0)
            cost += combined.cache_write_tokens * self.pricing.get("cache_write", 0.0)
            lines.append(f"Estimated cost:  ${cost / 1_000_000:.4f}")
        else:
            lines.append(
                'Estimated cost:  unavailable (set agent.pricing = {"input": ..., "output": ...} in $ per million tokens).'
            )
        return "\n".join(lines)

    async def _aexecute_command(self, command: str, args: str = "") -> str:
        if command == "/compact":
            record = await self.history.acompact()
            self.total_usage.add(self.history.drain_usage())
            return self._compact_report(record)
        return self._execute_command(command, args)

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

    @tracer.tool(name="Tool Call")
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
        ref = f"out_{uuid.uuid4().hex[:8]}"
        self._tool_outputs[ref] = content
        while len(self._tool_outputs) > MAX_STORED_TOOL_OUTPUTS:
            self._tool_outputs.pop(next(iter(self._tool_outputs)))
        head = keep_tokens_from_text(content, token_limit=limit, tokenizer_fn=tokenizer_fn)
        return (
            f"{head}\n[Output truncated: {total} tokens in total, {len(content)} characters. "
            f"Read the rest with {READ_TOOL_OUTPUT}(ref='{ref}', offset={len(head)}).]"
        )

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
            result = tool.run(**arguments)
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

    def execute_tool_calls(self, tool_calls: List[ToolCall]) -> List[Message]:
        results = []
        for tool_call in tool_calls:
            result = self.run_tool(tool_call)
            if result:
                results.append(result)

        return results

    async def aexecute_tool_calls(self, tool_calls: List[ToolCall]) -> List[Message]:
        tasks = []
        for tool_call in tool_calls:
            task = asyncio.create_task(asyncio.to_thread(self.run_tool, tool_call))
            tasks.append(task)

        results = await asyncio.gather(*tasks)

        return results

    def _pending_tool_calls(self) -> List[ToolCall]:
        """Tool calls of the last assistant turn that have no result yet.

        Read from the history alone: a run paused on approvals keeps no other
        state, so it can be resumed after a /save + /load or in another process.
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
        return [self._approval_request(tool_call) for tool_call in self._pending_tool_calls()]

    def _split_tool_calls(self, tool_calls: List[ToolCall]) -> tuple[List[ToolCall], List[ToolCall]]:
        """Separate the tool calls to run now from those that wait for the user's approval."""
        gated = {tool.name for tool in self.get_all_tools() if tool.requires_approval}
        return [t for t in tool_calls if t.name not in gated], [t for t in tool_calls if t.name in gated]

    def _decide_pending(self, approvals: Optional[ToolApprovals]) -> tuple[List[ToolCall], List[Message]]:
        """Apply the user's decisions to the pending tool calls.

        Returns the approved calls, to execute, and the tool results of the
        denied ones. A pending call left without a decision is denied too: the
        user moved on without answering, and the providers reject a history
        where a tool call has no result.
        """
        approvals = approvals or {}
        pending = self._pending_tool_calls()
        unknown = set(approvals) - {tool_call.id for tool_call in pending}
        if unknown:
            logger.warning(f"Ignoring approvals for tool calls that are not pending: {sorted(unknown)}")

        approved: List[ToolCall] = []
        denials: List[Message] = []
        for tool_call in pending:
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
        tool_messages = self.execute_tool_calls(tool_calls=tool_calls)
        agent_response.usage.tool_calls += len(tool_messages)
        yield from self._record_tool_results(tool_messages, agent_response)
        if self.team:
            # Fold the sub-agents' spend into this run's usage.
            agent_response.usage.add(self._drain_delegation_usage())

    async def _arun_tool_round(
        self, tool_calls: List[ToolCall], agent_response: AgentResponse
    ) -> AsyncIterator[AgentResponseEvent]:
        """Async twin of `_run_tool_round`: the calls of the round run in parallel."""
        if not tool_calls:
            return
        for event in self._start_tool_calls(tool_calls, agent_response):
            yield event
        tool_messages = await self.aexecute_tool_calls(tool_calls=tool_calls)
        agent_response.usage.tool_calls += len(tool_messages)
        for event in self._record_tool_results(tool_messages, agent_response):
            yield event
        if self.team:
            agent_response.usage.add(self._drain_delegation_usage())

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
        self, approvals: Optional[ToolApprovals], agent_response: AgentResponse
    ) -> Iterator[AgentResponseEvent]:
        """Settle the tool calls a previous run paused on, before new input reaches the history."""
        approved, denials = self._decide_pending(approvals)
        yield from self._run_tool_round(approved, agent_response)
        yield from self._record_tool_results(denials, agent_response)

    async def _aresume_tool_calls(
        self, approvals: Optional[ToolApprovals], agent_response: AgentResponse
    ) -> AsyncIterator[AgentResponseEvent]:
        approved, denials = self._decide_pending(approvals)
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
        messages, task = self._run_input(input)

        command = self._match_command(task)
        if command:
            return self._command_response(self._execute_command(*command))

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
        if messages:
            self._pin_task(messages)
        for _ in self._resume_tool_calls(approvals, agent_response):
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

        while num_turns_available > 0:
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
            agent_response.content = response.content or None
        elif not completed:
            agent_response.content = None
            logger.warning(f"Agent stopped after max_turns={self.max_turns} without a final answer.")

        agent_response.events.append(
            AgentEventCompleted(
                agent_id=self.id,
                run_id=agent_response.run_id,
            )
        )

        agent_response.finish_reason = "stop" if completed else "approval_required" if paused else "max_turns"
        drain = getattr(self.history, "drain_usage", None)
        if drain is not None:
            agent_response.usage.add(drain())
        self.total_usage.add(agent_response.usage)

        return agent_response

    @tracer.agent(name="Agent Arun")
    async def arun(
        self,
        input: Union[str, List[Message], List[Dict[str, str]], None] = None,
        context: Optional[Union[str, Dict[str, str]]] = None,
        approvals: Optional[ToolApprovals] = None,
    ) -> AgentResponse:
        messages, task = self._run_input(input)

        command = self._match_command(task)
        if command:
            return self._command_response(await self._aexecute_command(*command))

        agent_response = AgentResponse(agent_id=self.id)
        agent_response.events.append(
            AgentEventStarted(
                agent_id=self.id,
                run_id=agent_response.run_id,
            )
        )

        self._prepare_history(task=task, context=context)
        if messages:
            self._pin_task(messages)
        async for _ in self._aresume_tool_calls(approvals, agent_response):
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

        while num_turns_available > 0:
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
            agent_response.content = response.content or None
        elif not completed:
            agent_response.content = None
            logger.warning(f"Agent stopped after max_turns={self.max_turns} without a final answer.")

        agent_response.events.append(
            AgentEventCompleted(
                agent_id=self.id,
                run_id=agent_response.run_id,
            )
        )

        agent_response.finish_reason = "stop" if completed else "approval_required" if paused else "max_turns"
        drain = getattr(self.history, "drain_usage", None)
        if drain is not None:
            agent_response.usage.add(drain())
        self.total_usage.add(agent_response.usage)

        return agent_response

    @tracer.agent(name="Agent Stream")
    def run_stream(
        self,
        input: Union[str, List[Message], List[Dict[str, str]], None] = None,
        context: Optional[Union[str, Dict[str, str]]] = None,
        approvals: Optional[ToolApprovals] = None,
    ) -> Iterator[AgentResponse]:
        messages, task = self._run_input(input)

        command = self._match_command(task)
        if command:
            agent_response = self._command_response(self._execute_command(*command))
            for event in agent_response.events:
                yield event
            yield agent_response
            return

        agent_response = AgentResponse(agent_id=self.id)
        event = AgentEventStarted(
            agent_id=self.id,
            run_id=agent_response.run_id,
        )
        agent_response.events.append(event)
        yield event

        self._prepare_history(task=task, context=context)
        if messages:
            self._pin_task(messages)
        yield from self._resume_tool_calls(approvals, agent_response)
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

        while num_turns_available > 0:
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

        event = AgentEventCompleted(
            agent_id=self.id,
            run_id=agent_response.run_id,
        )
        agent_response.events.append(event)
        yield event

        agent_response.finish_reason = "stop" if completed else "approval_required" if paused else "max_turns"
        drain = getattr(self.history, "drain_usage", None)
        if drain is not None:
            agent_response.usage.add(drain())
        self.total_usage.add(agent_response.usage)

        yield agent_response

    @tracer.agent(name="Agent Astream")
    async def arun_stream(
        self,
        input: Union[str, List[Message], List[Dict[str, str]], None] = None,
        context: Optional[Union[str, Dict[str, str]]] = None,
        approvals: Optional[ToolApprovals] = None,
    ) -> AsyncIterator[AgentResponse]:
        messages, task = self._run_input(input)

        command = self._match_command(task)
        if command:
            agent_response = self._command_response(await self._aexecute_command(*command))
            for event in agent_response.events:
                yield event
            yield agent_response
            return

        agent_response = AgentResponse(agent_id=self.id)
        event = AgentEventStarted(
            agent_id=self.id,
            run_id=agent_response.run_id,
        )
        agent_response.events.append(event)
        yield event

        self._prepare_history(task=task, context=context)
        if messages:
            self._pin_task(messages)
        async for event in self._aresume_tool_calls(approvals, agent_response):
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

        while num_turns_available > 0:
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

        event = AgentEventCompleted(
            agent_id=self.id,
            run_id=agent_response.run_id,
        )
        agent_response.events.append(event)
        yield event

        agent_response.finish_reason = "stop" if completed else "approval_required" if paused else "max_turns"
        drain = getattr(self.history, "drain_usage", None)
        if drain is not None:
            agent_response.usage.add(drain())
        self.total_usage.add(agent_response.usage)

        yield agent_response
