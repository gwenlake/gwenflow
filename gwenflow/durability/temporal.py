"""`TemporalDurability`: makes an `Agent` or a `Flow` durable when it runs inside a Temporal workflow.

Outside a workflow it changes nothing. Inside one, the agent's model requests,
tool calls and summarizing compaction, or the flow's nodes, become activities;
the agent loop and the DAG walk stay in workflow code.

    agent = Agent(name="finance", llm=..., tools=[...], durability=TemporalDurability())
    flow = Flow.from_file("news.yaml", durability=TemporalDurability())
"""

from __future__ import annotations

import asyncio
import copy
import threading
import weakref
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from pydantic import TypeAdapter
from temporalio import activity, workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ApplicationError

from gwenflow.durability.agents import (
    _RESPONSE,
    _ActivityChat,
    _ActivityTool,
    _MemoryTool,
    _query_text,
    _RecordedReads,
)
from gwenflow.durability.errors import GwenflowDurabilityError
from gwenflow.durability.flows import FLOW_NODE_ACTIVITY, flow_node, node_activity_config
from gwenflow.flows.flows import Flow, Node
from gwenflow.memory.long_term.toolset import FlushReport, MemoryState
from gwenflow.stores.files.base import MemoryFile
from gwenflow.tools import BaseTool, ToolReturn
from gwenflow.tools.mcp import _MCPTool

_MEMORY_STATE = TypeAdapter(MemoryState)
_MEMORY_FILE = TypeAdapter(MemoryFile)
_FLUSH_REPORT = TypeAdapter(FlushReport)

DEFAULT_ACTIVITY_CONFIG: workflow.ActivityConfig = {"start_to_close_timeout": timedelta(seconds=60)}
DEFAULT_MODEL_ACTIVITY_CONFIG: workflow.ActivityConfig = {
    "start_to_close_timeout": timedelta(minutes=5),
    "heartbeat_timeout": timedelta(seconds=30),
}

# Temporal's own default retries an activity forever. Bounded defaults keep a run
# from hanging: a tool that keeps failing reaches the model as an error message
# (as outside Temporal), a model request that keeps failing fails the run.
DEFAULT_RETRY_POLICY = RetryPolicy(maximum_attempts=3, maximum_interval=timedelta(seconds=30))
# 10 attempts spread over about 4 minutes: rides out a rate limit or a short provider outage.
DEFAULT_MODEL_RETRY_POLICY = RetryPolicy(maximum_attempts=10, maximum_interval=timedelta(seconds=60))

# Client errors a retry cannot fix (bad request, auth, not found, too large...).
# 408, 409 and 429 are transient and stay retryable.
_RETRYABLE_CLIENT_STATUSES = {408, 409, 429}


# Without a configured heartbeat timeout, how often a long activity reports it is alive.
HEARTBEAT_EVERY = 10.0


def _http_status(error: BaseException) -> Optional[int]:
    # `status_code`: openai, anthropic, mistralai. `code`: google-genai (openai also has a
    # `code`, but it is a string such as "invalid_api_key").
    for attribute in ("status_code", "code"):
        status = getattr(error, attribute, None)
        if isinstance(status, int) and not isinstance(status, bool) and 100 <= status < 600:
            return status
    return None


def _status_error(error: BaseException) -> Optional[BaseException]:
    """The exception that carries an HTTP status: `error` itself, or one of its causes.

    The provider adapters wrap the SDK error (`RuntimeError(...) from e`), so the
    status is looked for along the chain of causes; the first one found decides.
    """
    seen: set[int] = set()
    source: Optional[BaseException] = error
    while source is not None and id(source) not in seen:
        seen.add(id(source))
        if _http_status(source) is not None:
            return source
        source = source.__cause__
    return None


def _non_retryable_status(error: BaseException) -> Optional[int]:
    """The HTTP status of a provider SDK error when retrying it is pointless, else None."""
    source = _status_error(error)
    status = None if source is None else _http_status(source)
    if status is not None and 400 <= status < 500 and status not in _RETRYABLE_CLIENT_STATUSES:
        return status
    return None  # no status (network...), 5xx, 408, 409, 429: Temporal retries it


async def _alive(awaitable: Any) -> Any:
    """Await `awaitable` in an activity, sending heartbeats until it is done.

    A model request or a compaction can outlast the heartbeat timeout, and the
    server takes a silent activity for a dead worker. Heartbeats go out three
    times per timeout, as for flow nodes.
    """
    task = asyncio.ensure_future(awaitable)
    if not activity.in_activity():
        return await task
    timeout = activity.info().heartbeat_timeout
    every = timeout.total_seconds() / 3 if timeout else HEARTBEAT_EVERY
    try:
        while not task.done():
            activity.heartbeat()
            await asyncio.wait({task}, timeout=every)
    except asyncio.CancelledError:
        task.cancel()
        raise
    return task.result()


async def _model_response(llm: Any, messages: Any) -> Dict[str, Any]:
    """One request to `llm` from an activity: kept alive, and failed at once when rejected."""
    try:
        response = await _alive(llm.ainvoke(input=messages))
    except Exception as e:
        status = _non_retryable_status(e)
        if status is None:
            raise  # transient or unknown: Temporal retries it
        raise ApplicationError(
            f"Model request rejected ({status}): {e}", type=type(_status_error(e)).__name__, non_retryable=True
        ) from e
    return _RESPONSE.dump_python(response, mode="json")


@dataclass
class TemporalDurability:
    activity_config: workflow.ActivityConfig = field(default_factory=lambda: dict(DEFAULT_ACTIVITY_CONFIG))
    """Base config for every activity. Without a `retry_policy`, an activity gets at most
    3 attempts (a tool: its own `max_retries`, 1 if it requires approval)."""

    model_activity_config: workflow.ActivityConfig = field(default_factory=lambda: dict(DEFAULT_MODEL_ACTIVITY_CONFIG))
    """Merged over `activity_config` for model requests (agent llm and summarizer)."""

    tool_activity_config: Dict[str, workflow.ActivityConfig] = field(default_factory=dict)
    """Per-tool overrides, keyed by tool name, merged over `activity_config`."""

    continue_as_new: Optional[Callable[[Any], None]] = None
    """Called with the workflow-side agent before a model request when Temporal
    suggests continuing as new. It should call `workflow.continue_as_new(...)`
    with what the workflow needs to resume (e.g. `[m.to_dict() for m in agent.history.messages]`)."""

    _target: Any = field(default=None, init=False, repr=False)
    _activities: List[Callable] = field(default_factory=list, init=False, repr=False)
    _copies: Any = field(default_factory=weakref.WeakKeyDictionary, init=False, repr=False)
    _mcp_tools: Dict[Tuple[str, str], BaseTool] = field(default_factory=dict, init=False, repr=False)
    _mcp_lock: Any = field(default_factory=threading.Lock, init=False, repr=False)

    # -- binding (called from Agent.__post_init__) --------------------------

    def bind(self, target: Any) -> None:
        """Called by `Agent` / `Flow` at construction. One instance per agent or flow."""
        if self._target is not None and self._target is not target:
            raise GwenflowDurabilityError("A TemporalDurability serves a single agent or flow: create one for each.")
        if isinstance(target, Flow):
            self._target = target
            self._activities = [flow_node]
            return
        agent = target
        if not agent.name or not agent.name.replace("-", "_").isidentifier():
            raise GwenflowDurabilityError(
                "A durable agent needs a stable `name` (letters, digits, '_' or '-'): "
                "activity names are derived from it and must not change once deployed."
            )
        self._target = agent
        self._activities = self._build_activities()
        # The tokenizer loads its vocabulary lazily (disk cache, network on first use):
        # do it now, outside any workflow, so token counting stays pure in workflow code.
        agent.history.tokenizer_fn("warm-up")

    def check_supported(self) -> None:
        """Raise if the bound agent cannot run inside a workflow.

        Not checked when the agent is built: an agent with a durability runs
        unchanged outside a workflow, whatever its options. `worker_options()`
        calls this when the worker starts, and a workflow run that still
        reaches such an agent fails at once instead of hanging.
        """
        agent = self._target
        if agent is None or isinstance(agent, Flow):
            return
        problems = []
        unsupported = [label for label, present in (("team", agent.team), ("subagents", agent.subagents)) if present]
        if unsupported:
            problems.append(f"not supported yet with Temporal: {', '.join(unsupported)}")
        server_names = [server.name for server in agent.mcp_servers]
        duplicates = sorted({n for n in server_names if server_names.count(n) > 1})
        if duplicates:
            problems.append(f"its MCP servers need distinct names (pass `name=`): {', '.join(duplicates)}")
        for label, llm in (("llm", agent.llm), ("reasoning_model", agent.reasoning_model)):
            if llm is not None and llm.context_window == "auto":
                problems.append(f"{label} has context_window='auto', which queries the server: set an int")
        if problems:
            raise GwenflowDurabilityError(f"Agent '{agent.name}' cannot run in a workflow: {'; '.join(problems)}.")

    @property
    def activities(self) -> List[Callable]:
        return list(self._activities)

    def _activity_name(self, kind: str) -> str:
        return f"gwenflow__{self._target.name}__{kind}"

    def _config(self) -> workflow.ActivityConfig:
        return {"retry_policy": DEFAULT_RETRY_POLICY, **self.activity_config}

    def _config_for_model(self) -> workflow.ActivityConfig:
        return {"retry_policy": DEFAULT_MODEL_RETRY_POLICY, **self.activity_config, **self.model_activity_config}

    def _config_for_tool(self, tool_name: str, max_attempts: Optional[int] = None) -> workflow.ActivityConfig:
        # The tool's own attempt count (1 for a tool that requires approval) replaces
        # its in-process retries, which do not run inside the activity.
        retry_policy = (
            DEFAULT_RETRY_POLICY
            if max_attempts is None
            else RetryPolicy(maximum_attempts=max(max_attempts, 1), maximum_interval=timedelta(seconds=30))
        )
        return {"retry_policy": retry_policy, **self.activity_config, **self.tool_activity_config.get(tool_name, {})}

    # -- worker side: the activities that do the real I/O --------------------

    def _build_activities(self) -> List[Callable]:
        durability = self

        @activity.defn(name=self._activity_name("model_request"))
        async def model_request(request: Dict[str, Any]) -> Dict[str, Any]:
            agent = durability._target
            # One llm copy per call: activities of concurrent workflows share this worker-side agent.
            llm = copy.copy(agent.llm)
            tools_by_name = {t.name: t for t in agent.tools}
            if agent.response_model is not None:
                final_result = agent._final_result_tool()
                tools_by_name[final_result.name] = final_result
            for d in request.get("mcp_tools") or []:
                tools_by_name[d["name"]] = _MCPTool(
                    _mcp_name=d["name"], _mcp_description=d["description"], _mcp_parameters=d["parameters"]
                )
            names = request["tool_names"]
            llm.tools = [tools_by_name[n] for n in names] if names else None
            llm.tool_choice = request["tool_choice"]
            if hasattr(llm, "response_format"):
                llm.response_format = agent.response_model if request["structured_output"] else None
            return await _model_response(llm, request["messages"])

        @activity.defn(name=self._activity_name("reasoning_request"))
        async def reasoning_request(request: Dict[str, Any]) -> Dict[str, Any]:
            # The reasoning step: one request to `reasoning_model`, without tools.
            llm = copy.copy(durability._target.reasoning_model)
            llm.tools = None
            return await _model_response(llm, request["messages"])

        @activity.defn(name=self._activity_name("call_tool"))
        async def call_tool(request: Dict[str, Any]) -> Union[str, Dict[str, Any], None]:
            server = request.get("server")
            if server is None:
                tool = {t.name: t for t in durability._target.tools}[request["name"]]
            else:
                tool = await asyncio.to_thread(durability._mcp_tool, server, request["name"])
            # A single attempt: retries belong to Temporal, not to tenacity inside the activity.
            result = tool._cast_result(await asyncio.to_thread(tool._run, **request["arguments"]))
            if isinstance(result, ToolReturn):
                # Its metadata must be serializable by the workflow's data converter.
                return {"return_value": result.return_value, "metadata": result.metadata}
            return result

        @activity.defn(name=self._activity_name("compact_history"))
        async def compact_history(request: Dict[str, Any]) -> Dict[str, Any]:
            # The whole compaction runs here: it is a pure function of the history plus
            # summarizer calls, so the workflow only records its result.
            history = durability._target._fresh_history()
            history.context_window = request["context_window"]
            history.system_prompt = request["system_prompt"]
            for message in request["messages"]:
                history.add_message(message)
            if hasattr(history, "summary"):
                history.summary = request["summary"] or ""
                history._summarized_upto = request["summarized_upto"]
            record = await _alive(asyncio.to_thread(history.compact))
            return {
                "changed": record is not None,
                "messages": [m.to_dict() for m in history.messages],
                "summary": getattr(history, "summary", None),
                "summarized_upto": getattr(history, "_summarized_upto", 0),
            }

        @activity.defn(name=self._activity_name("list_mcp_tools"))
        async def list_mcp_tools(request: Dict[str, Any]) -> List[Dict[str, Any]]:
            listing = await asyncio.to_thread(durability._list_mcp_tools)
            return [
                {"server": server, "name": t.name, "description": t.description or "", "parameters": t.parameters}
                for server, t in listing
            ]

        @activity.defn(name=self._activity_name("memory_load"))
        async def memory_load(request: Dict[str, Any]) -> Dict[str, Any]:
            memory = durability._target.memory.fresh_copy()
            await memory.aload(query=request.get("query"))
            return _MEMORY_STATE.dump_python(memory.dump_state(), mode="json")

        @activity.defn(name=self._activity_name("memory_read"))
        async def memory_read(request: Dict[str, Any]) -> Dict[str, Any]:
            store = durability._target.memory.store
            files = {}
            for path in request["paths"]:
                file = await store.aread(path)
                files[path] = None if file is None else _MEMORY_FILE.dump_python(file, mode="json")
            return files

        @activity.defn(name=self._activity_name("memory_flush"))
        async def memory_flush(request: Dict[str, Any]) -> Dict[str, Any]:
            memory = durability._target.memory.fresh_copy()
            memory.restore_state(_MEMORY_STATE.validate_python(request))
            report = await memory.aflush()
            if report.errors:
                raise ApplicationError(f"Memory flush failed: {report.errors}", type="MemoryFlushError")
            return {
                "state": _MEMORY_STATE.dump_python(memory.dump_state(), mode="json"),
                "report": _FLUSH_REPORT.dump_python(report, mode="json"),
            }

        activities = [model_request, call_tool, compact_history]
        if self._target.reasoning_model is not None:
            activities.append(reasoning_request)
        if self._target.mcp_servers:
            activities.append(list_mcp_tools)
        if self._target.memory is not None:
            activities.extend([memory_load, memory_read, memory_flush])
        return activities

    def _list_mcp_tools(self) -> List[Tuple[str, BaseTool]]:
        with self._mcp_lock:
            listing = []
            for server in self._target.mcp_servers:
                for tool in server.get_tools():
                    self._mcp_tools[(server.name, tool.name)] = tool
                    listing.append((server.name, tool))
            return listing

    def _mcp_tool(self, server: str, name: str) -> BaseTool:
        tool = self._mcp_tools.get((server, name))
        if tool is None:
            self._list_mcp_tools()
            tool = self._mcp_tools.get((server, name))
        if tool is None:
            raise ApplicationError(
                f"MCP tool {name!r} is no longer offered by server {server!r}.",
                type="UnknownMCPTool",
                non_retryable=True,
            )
        return tool

    async def _compact_in_activity(self, history: Any) -> None:
        data = await workflow.execute_activity(
            self._activity_name("compact_history"),
            {
                "messages": [m.to_dict() for m in history.messages],
                "summary": getattr(history, "summary", None),
                "summarized_upto": getattr(history, "_summarized_upto", 0),
                "context_window": history.context_window,
                "system_prompt": history.system_prompt,
            },
            **self._config_for_model(),
        )
        if not data["changed"]:
            return None
        system_prompt = history.system_prompt
        history.reset()
        history.system_prompt = system_prompt
        for message in data["messages"]:
            history.add_message(message)
        if hasattr(history, "summary"):
            history.summary = data["summary"] or ""
            history._summarized_upto = data["summarized_upto"]
        return None

    async def before_run(self, agent: Any, input: Any = None) -> None:
        if getattr(agent, "_mcp_pending", False):
            listing = await workflow.execute_activity(self._activity_name("list_mcp_tools"), {}, **self._config())
            agent.tools = list(agent.tools) + [_ActivityTool.for_mcp(d, self) for d in listing]
            agent.llm.tools = agent._llm_tools() or None
            agent._mcp_pending = False
        if agent.memory is not None and (input is not None or not agent.memory.loaded):
            state = await workflow.execute_activity(
                self._activity_name("memory_load"), {"query": _query_text(input)}, **self._config()
            )
            agent.memory.store.files.clear()
            agent.memory.restore_state(_MEMORY_STATE.validate_python(state))

    async def after_run(self, agent: Any) -> None:
        if agent.memory is None or not agent.memory.has_pending():
            return
        result = await workflow.execute_activity(
            self._activity_name("memory_flush"),
            _MEMORY_STATE.dump_python(agent.memory.dump_state(pending_only=True), mode="json"),
            **self._config(),
        )
        agent.memory.merge_flushed(_MEMORY_STATE.validate_python(result["state"]))

    async def _fetch_memory_files(self, memory: Any, paths: List[str]) -> None:
        missing = [p for p in memory.unloaded(paths) if p not in memory.store.files]
        if not missing:
            return
        files = await workflow.execute_activity(
            self._activity_name("memory_read"), {"paths": missing}, **self._config()
        )
        for path, data in files.items():
            memory.store.files[path] = None if data is None else _MEMORY_FILE.validate_python(data)

    def before_model_request(self, agent: Any) -> None:
        """Hook called by the agent loop before each model request, in workflow code."""
        if getattr(agent, "_durable_helper", False):
            return  # the reasoning step's own agent: nothing to carry over
        if self.continue_as_new is not None and workflow.info().is_continue_as_new_suggested():
            self.continue_as_new(agent)

    # -- workflow side: the proxies --------------------------------------------

    # -- workflow side ---------------------------------------------------------

    def workflow_agent(self, agent: Any) -> Any:
        """The agent's copy for the current workflow run, created on first use.

        Same configuration, its own history, llm and tools proxied to activities.
        Every `agent.arun()` of the run reuses it, so a multi-turn conversation
        keeps its history; a replay rebuilds it identically.
        """
        instance = workflow.instance()
        clone = self._copies.get(instance)
        if clone is not None:
            return clone
        try:
            self.check_supported()
        except GwenflowDurabilityError as e:
            # A configuration error: fail the run, a retried workflow task would hang on it.
            raise ApplicationError(str(e), type="UnsupportedAgent", non_retryable=True) from e
        clone = agent.clone()
        clone._workflow_copy = True
        clone._mcp_pending = bool(agent.mcp_servers)
        clone.mcp_servers = []
        memory_tools = set()
        if agent.memory is not None:
            clone.memory = agent.memory.fresh_copy(store=_RecordedReads())
            memory_tools = {
                id(t) for t in agent.tools if getattr(getattr(t, "function", None), "__self__", None) is agent.memory
            }
        clone.tools = [
            _MemoryTool.wrap(t, clone.memory, self) if id(t) in memory_tools else _ActivityTool.wrap(t, self)
            for t in agent.tools
        ]
        proxy = _ActivityChat.wrap(agent.llm, self)
        proxy.tools = clone._llm_tools() if clone.tools else None
        proxy.tool_choice = agent.llm.tool_choice
        proxy.response_format = getattr(agent.llm, "response_format", None)
        clone.llm = proxy
        if agent.reasoning_model is not None:
            clone.reasoning_model = _ActivityChat.wrap(agent.reasoning_model, self, activity="reasoning_request")
        # Over-budget compaction (the one that calls the summarizer) becomes an activity.
        clone.history.durable_compactor = self._compact_in_activity
        self._copies[instance] = clone
        return clone

    def adopt(self, helper: Any) -> None:
        """Make the agent of the reasoning step durable: its llm is already the activity proxy."""
        helper.durability = self
        helper._workflow_copy = True
        helper._durable_helper = True
        helper.history.durable_compactor = self._compact_in_activity

    async def execute_node(self, node: Node, input_data: Optional[Dict[str, Any]]) -> Any:
        """Flow side: run one node of the bound flow as a `gwenflow__flow_node` activity."""
        return await workflow.execute_activity(
            FLOW_NODE_ACTIVITY,
            {"node": asdict(node), "input_data": input_data},
            **node_activity_config(node, self._target),
        )
