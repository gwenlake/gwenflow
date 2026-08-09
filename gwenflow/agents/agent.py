import asyncio
import copy
import json
import re
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, Iterator, List, Literal, Optional, Type, Union

from pydantic import BaseModel, ValidationError

from gwenflow.agents.prompts import PROMPT_CONTEXT, PROMPT_JSON_SCHEMA, PROMPT_KNOWLEDGE
from gwenflow.llms import ChatBase, ChatOpenAI
from gwenflow.logger import logger
from gwenflow.memory import ChatMemoryBuffer, ChatSummaryMemoryBuffer
from gwenflow.retriever import Retriever
from gwenflow.skills import Skill, SkillsToolset
from gwenflow.telemetry import tracer
from gwenflow.tools import BaseTool, MCPServer, Tool
from gwenflow.types import (
    AgentEventCompleted,
    AgentEventContent,
    AgentEventStarted,
    AgentEventThinking,
    AgentEventToolCompleted,
    AgentEventToolStarted,
    AgentResponse,
    AgentUsage,
    Message,
    ModelResponse,
    RequestUsage,
    ToolCall,
    ToolResponse,
)
from gwenflow.utils import extract_json_str

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

    skills: List[Skill] = field(default_factory=list)
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
            # The agent configures tools/tool_choice/response_format on its
            # LLM below: take a private copy so that sharing one llm instance
            # between several agents (orchestrator + team members) does not
            # cross-contaminate their tool lists. The underlying HTTP client
            # is still shared (shallow copy).
            self.llm = copy.copy(self.llm)
        if self.history is None:
            # Dedicated summarizer: same model, but stripped of the agent's
            # tools and response_format (assigned on self.llm below) so the
            # compaction calls stay plain text-in / text-out.
            summarizer = copy.copy(self.llm)
            summarizer.tools = []
            summarizer.tool_choice = None
            summarizer.response_format = None
            self.history = ChatSummaryMemoryBuffer(
                token_limit=self.llm.get_context_size(),
                llm=summarizer,
            )
        if self.response_model:
            self.llm.response_format = self.response_model
        if self.skills:
            self.tools = list(self.tools) + SkillsToolset(self.skills).get_tools()
        if self.team:
            self._handoff_tools = self._build_handoff_tools()
            self.tools = list(self.tools) + self._handoff_tools
        if self.tools or self.mcp_servers:
            self.llm.tools = self.get_all_tools()
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
        usage.add(response.usage)
        compaction = getattr(worker.history, "usage", None)
        if compaction is not None and compaction.requests:
            usage.add(compaction)
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

    def _compact_report(self, folded_before: int) -> str:
        folded = self.history._summarized_upto - folded_before
        if folded == 0:
            return "Nothing to compact: the conversation fits in the context window."
        summary_tokens = self.history.tokenizer_fn(self.history.summary)
        return f"Compacted {folded} message(s) into the running summary (~{summary_tokens} tokens)."

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
            if not hasattr(self.history, "_summarized_upto"):
                return "This agent's memory does not support compaction (plain ChatMemoryBuffer)."
            before = self.history._summarized_upto
            self.history.compact()
            return self._compact_report(before)
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
            # Knowledge references are query-dependent: resolve without the retriever.
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
            parts = [f"{len(messages)} message(s) stored, ~{tokens} tokens (limit {self.history.token_limit})."]
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
        if self.team:
            lines.append(f"  team:            {', '.join(m.name or m.id for m in self.team)}")
        if self.response_model is not None:
            model_name = self.response_model.__name__ if isinstance(self.response_model, type) else "dict schema"
            lines.append(f"  response_model:  {model_name}")
        lines.append(
            f"  memory:          {type(self.history).__name__} "
            f"(token_limit {self.history.token_limit:,}, reserved {self.history.reserved_tokens:,})"
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
            self.history._summarized_upto = min(int(data.get("summarized_upto") or 0), len(messages))
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
            combined.add(compaction_usage)
            lines += block("Compaction:", compaction_usage)

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
        if command == "/compact" and hasattr(self.history, "_summarized_upto"):
            before = self.history._summarized_upto
            await self.history.acompact()
            return self._compact_report(before)
        return self._execute_command(command, args)

    def tool(self, func: Callable) -> Callable:
        """Decorator that registers a function as a tool on this agent instance."""
        # self.tools = list(self.tools) + [FunctionTool.from_function(func)]
        self.tools = list(self.tools) + [Tool(func)]
        self.llm.tools = self.get_all_tools()
        self.llm.tool_choice = self.tool_choice
        return func

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
        self.history.system_prompt = self.get_system_prompt(task=task, context=context)
        self.history.reserved_tokens = self._tools_token_count()

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
    def run_tool(self, tool_call: ToolCall) -> Message:
        tool_execution = ToolResponse(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
        )

        tool_map = {tool.name: tool for tool in self.get_all_tools()}

        if tool_call.name not in tool_map.keys():
            logger.error(f"Tool {tool_call.name} does not exist")
            tool_execution.content = f"Tool {tool_call.name} does not exist"
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
            # Include the exception detail so the model can self-correct
            # (wrong argument name, malformed JSON, ...).
            tool_execution.content = f"Error executing tool '{tool_call.name}': {e}"
            return tool_execution.to_message()

        # A falsy result ("", 0, False, [], None) is a valid tool outcome,
        # not an error. ToolResponse.content is the *stringified* value.
        if result is None:
            tool_execution.content = ""
        elif isinstance(result, str):
            tool_execution.content = result
        else:
            tool_execution.content = str(result)
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

    @tracer.agent(name="Agent Run")
    def run(
        self,
        input: Union[str, List[Message], List[Dict[str, str]]],
        context: Optional[Union[str, Dict[str, str]]] = None,
    ) -> AgentResponse:
        # prepare messages and task
        messages = self.llm.input_to_message_list(input)
        task = messages[-1].content

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
        self.history.add_messages(messages)

        # add reasoning
        if self.reasoning_model:
            self.history.compact()
            messages_for_reasoning_model = [m.to_dict() for m in self.history.get()]
            response = self.reason(messages_for_reasoning_model)
            if response is not None:  # reason() returns None on an empty plan
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
        completed = False

        while num_turns_available > 0:
            num_turns_available -= 1

            # format messages
            self.history.compact()
            messages_for_model = [m.to_dict() for m in self.history.get()]

            # call llm and tool
            response = self.llm.invoke(input=messages_for_model)

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

            # stop if not tool call
            if not response.tool_calls:
                content_to_check = response.parsed if response.parsed else response.content
                is_valid, parsed_data, error_msg = self._validate_final_response(content_to_check)

                if not is_valid and validation_retries_left > 0:
                    validation_retries_left -= 1
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

            # handle tool calls
            if response.tool_calls and self.get_all_tools():
                for tool_call in response.tool_calls:
                    args = self._parse_tool_args(tool_call)
                    agent_response.events.append(
                        AgentEventToolStarted(
                            agent_id=self.id,
                            run_id=agent_response.run_id,
                            tool_call_id=tool_call.id,
                            tool_name=tool_call.name,
                            tool_args=args,
                        )
                    )

                tool_messages = self.execute_tool_calls(tool_calls=response.tool_calls)
                for m in tool_messages:
                    agent_response.usage.tool_calls += 1
                    self.history.add_message(m)
                    agent_response.messages.append(m)
                    agent_response.events.append(
                        AgentEventToolCompleted(
                            agent_id=self.id,
                            run_id=agent_response.run_id,
                            tool_call_id=m.tool_call_id,
                            content=m.content,
                        )
                    )
                if self.team:
                    # Fold the sub-agents' spend into this run's usage.
                    agent_response.usage.add(self._drain_delegation_usage())

            if self.tool_choice == "required":
                # The LLM reads its own tool_choice at request time: downgrade
                # it too, otherwise every turn keeps forcing a tool call and
                # the agent can never produce a final answer.
                self.tool_choice = "auto"
                self.llm.tool_choice = "auto"

        if not completed:
            # max_turns exhausted without a final answer: don't report the
            # partial state of the last (tool-calling) turn as a normal stop.
            agent_response.content = None
            logger.warning(f"Agent stopped after max_turns={self.max_turns} without a final answer.")

        agent_response.events.append(
            AgentEventCompleted(
                agent_id=self.id,
                run_id=agent_response.run_id,
            )
        )

        agent_response.finish_reason = "stop" if completed else "max_turns"
        self.total_usage.add(agent_response.usage)

        return agent_response

    @tracer.agent(name="Agent Arun")
    async def arun(
        self,
        input: Union[str, List[Message], List[Dict[str, str]]],
        context: Optional[Union[str, Dict[str, str]]] = None,
    ) -> AgentResponse:
        messages = self.llm.input_to_message_list(input)
        task = messages[-1].content

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
        self.history.add_messages(messages)

        if self.reasoning_model:
            await self.history.acompact()
            messages_for_reasoning_model = [m.to_dict() for m in self.history.get()]
            reasoning_agent_response = await self.areason(messages_for_reasoning_model)
            if reasoning_agent_response is not None:  # areason() returns None on an empty plan
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
        completed = False

        while num_turns_available > 0:
            num_turns_available -= 1

            await self.history.acompact()
            messages_for_model = [m.to_dict() for m in self.history.get()]

            response = await self.llm.ainvoke(input=messages_for_model)

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

            if not response.tool_calls:
                content_to_check = response.parsed if response.parsed else response.content
                is_valid, parsed_data, error_msg = self._validate_final_response(content_to_check)

                if not is_valid and validation_retries_left > 0:
                    validation_retries_left -= 1
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

            # handle tool calls
            if response.tool_calls and self.get_all_tools():
                for tool_call in response.tool_calls:
                    args = self._parse_tool_args(tool_call)
                    agent_response.events.append(
                        AgentEventToolStarted(
                            agent_id=self.id,
                            run_id=agent_response.run_id,
                            tool_call_id=tool_call.id,
                            tool_name=tool_call.name,
                            tool_args=args,
                        )
                    )

                tool_messages = await self.aexecute_tool_calls(tool_calls=response.tool_calls)
                for m in tool_messages:
                    agent_response.usage.tool_calls += 1
                    self.history.add_message(m)
                    agent_response.messages.append(m)
                    agent_response.events.append(
                        AgentEventToolCompleted(
                            agent_id=self.id,
                            run_id=agent_response.run_id,
                            tool_call_id=m.tool_call_id,
                            content=m.content,
                        )
                    )
                if self.team:
                    # Fold the sub-agents' spend into this run's usage.
                    agent_response.usage.add(self._drain_delegation_usage())

            if self.tool_choice == "required":
                # The LLM reads its own tool_choice at request time: downgrade
                # it too, otherwise every turn keeps forcing a tool call and
                # the agent can never produce a final answer.
                self.tool_choice = "auto"
                self.llm.tool_choice = "auto"

        if not completed:
            # max_turns exhausted without a final answer: don't report the
            # partial state of the last (tool-calling) turn as a normal stop.
            agent_response.content = None
            logger.warning(f"Agent stopped after max_turns={self.max_turns} without a final answer.")

        agent_response.events.append(
            AgentEventCompleted(
                agent_id=self.id,
                run_id=agent_response.run_id,
            )
        )

        agent_response.finish_reason = "stop" if completed else "max_turns"
        self.total_usage.add(agent_response.usage)

        return agent_response

    @tracer.agent(name="Agent Stream")
    def run_stream(
        self,
        input: Union[str, List[Message], List[Dict[str, str]]],
        context: Optional[Union[str, Dict[str, str]]] = None,
    ) -> Iterator[AgentResponse]:
        messages = self.llm.input_to_message_list(input)
        task = messages[-1].content

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
        self.history.add_messages(messages)

        if self.reasoning_model:
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
        completed = False

        while num_turns_available > 0:
            num_turns_available -= 1

            self.history.compact()
            messages_for_model = [m.to_dict() for m in self.history.get()]

            full_content = ""
            full_thinking = ""
            final_tool_calls: List[ToolCall] = []
            turn_usage = RequestUsage()
            # Accumulate raw thinking parts across the turn so we can re-derive
            # provider-shaped echo blocks (signatures etc.) at the end.
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

                # Stash raw parts (incl. ThinkingContent with signatures) for echo-back
                turn_response.parts.extend(chunk.parts)

            agent_response.usage.add(turn_usage)

            if full_thinking:
                if agent_response.reasoning_content:
                    agent_response.reasoning_content += "\n\n" + full_thinking
                else:
                    agent_response.reasoning_content = full_thinking

            tool_calls_dicts = [t.to_message_dict() for t in final_tool_calls]
            _message = Message(role="assistant", content=full_content, tool_calls=tool_calls_dicts)
            _message.thinking_parts = self.llm.get_thinking_parts(turn_response)
            self.history.add_message(_message)

            if not final_tool_calls:
                if self.response_model:
                    is_valid, parsed_data, error_msg = self._validate_final_response(full_content)
                    if not is_valid and validation_retries_left > 0:
                        validation_retries_left -= 1
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
                for tool_call in final_tool_calls:
                    args = self._parse_tool_args(tool_call)
                    event = AgentEventToolStarted(
                        agent_id=self.id,
                        run_id=agent_response.run_id,
                        tool_call_id=tool_call.id,
                        tool_name=tool_call.name,
                        tool_args=args,
                    )
                    agent_response.events.append(event)
                    yield event

                tool_messages = self.execute_tool_calls(tool_calls=final_tool_calls)
                for m in tool_messages:
                    agent_response.usage.tool_calls += 1
                    self.history.add_message(m)
                    agent_response.messages.append(m)
                    event = AgentEventToolCompleted(
                        agent_id=self.id, run_id=agent_response.run_id, tool_call_id=m.tool_call_id, content=m.content
                    )
                    agent_response.events.append(event)
                    yield event
                if self.team:
                    # Fold the sub-agents' spend into this run's usage.
                    agent_response.usage.add(self._drain_delegation_usage())

            if self.tool_choice == "required":
                # The LLM reads its own tool_choice at request time: downgrade
                # it too, otherwise every turn keeps forcing a tool call and
                # the agent can never produce a final answer.
                self.tool_choice = "auto"
                self.llm.tool_choice = "auto"

        if not completed:
            # max_turns exhausted without a final answer: don't leave the last
            # streamed delta as content nor report a normal stop.
            agent_response.content = None
            logger.warning(f"Agent stopped after max_turns={self.max_turns} without a final answer.")

        event = AgentEventCompleted(
            agent_id=self.id,
            run_id=agent_response.run_id,
        )
        agent_response.events.append(event)
        yield event

        agent_response.finish_reason = "stop" if completed else "max_turns"
        self.total_usage.add(agent_response.usage)

        yield agent_response

    @tracer.agent(name="Agent Astream")
    async def arun_stream(
        self,
        input: Union[str, List[Message], List[Dict[str, str]]],
        context: Optional[Union[str, Dict[str, str]]] = None,
    ) -> AsyncIterator[AgentResponse]:
        messages = self.llm.input_to_message_list(input)
        task = messages[-1].content

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
        self.history.add_messages(messages)

        if self.reasoning_model:
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
        completed = False

        while num_turns_available > 0:
            num_turns_available -= 1

            await self.history.acompact()
            messages_for_model = [m.to_dict() for m in self.history.get()]

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

            if full_thinking:
                if agent_response.reasoning_content:
                    agent_response.reasoning_content += "\n\n" + full_thinking
                else:
                    agent_response.reasoning_content = full_thinking

            tool_calls_dicts = [t.to_message_dict() for t in final_tool_calls]
            _message = Message(role="assistant", content=full_content, tool_calls=tool_calls_dicts)
            _message.thinking_parts = self.llm.get_thinking_parts(turn_response)
            self.history.add_message(_message)

            if not final_tool_calls:
                if self.response_model:
                    is_valid, parsed_data, error_msg = self._validate_final_response(full_content)
                    if not is_valid and validation_retries_left > 0:
                        validation_retries_left -= 1
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
                for tool_call in final_tool_calls:
                    args = self._parse_tool_args(tool_call)
                    event = AgentEventToolStarted(
                        agent_id=self.id,
                        run_id=agent_response.run_id,
                        tool_call_id=tool_call.id,
                        tool_name=tool_call.name,
                        tool_args=args,
                    )
                    agent_response.events.append(event)
                    yield event

                tool_messages = await self.aexecute_tool_calls(tool_calls=final_tool_calls)
                for m in tool_messages:
                    agent_response.usage.tool_calls += 1
                    self.history.add_message(m)
                    agent_response.messages.append(m)
                    event = AgentEventToolCompleted(
                        agent_id=self.id, run_id=agent_response.run_id, tool_call_id=m.tool_call_id, content=m.content
                    )
                    agent_response.events.append(event)
                    yield event
                if self.team:
                    # Fold the sub-agents' spend into this run's usage.
                    agent_response.usage.add(self._drain_delegation_usage())

            if self.tool_choice == "required":
                # The LLM reads its own tool_choice at request time: downgrade
                # it too, otherwise every turn keeps forcing a tool call and
                # the agent can never produce a final answer.
                self.tool_choice = "auto"
                self.llm.tool_choice = "auto"

        if not completed:
            # max_turns exhausted without a final answer: don't leave the last
            # streamed delta as content nor report a normal stop.
            agent_response.content = None
            logger.warning(f"Agent stopped after max_turns={self.max_turns} without a final answer.")

        event = AgentEventCompleted(
            agent_id=self.id,
            run_id=agent_response.run_id,
        )
        agent_response.events.append(event)
        yield event

        agent_response.finish_reason = "stop" if completed else "max_turns"
        self.total_usage.add(agent_response.usage)

        yield agent_response
