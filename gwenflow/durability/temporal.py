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
import weakref
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from typing import Any, Callable, Dict, List, Optional

from temporalio import activity, workflow

from gwenflow.durability.agents import _RESPONSE, _ActivityChat, _ActivityTool
from gwenflow.durability.errors import GwenflowDurabilityError
from gwenflow.durability.flows import FLOW_NODE_ACTIVITY, flow_node, node_activity_config
from gwenflow.flows.flows import Flow, Node

DEFAULT_ACTIVITY_CONFIG: workflow.ActivityConfig = {"start_to_close_timeout": timedelta(seconds=60)}
DEFAULT_MODEL_ACTIVITY_CONFIG: workflow.ActivityConfig = {
    "start_to_close_timeout": timedelta(minutes=5),
    "heartbeat_timeout": timedelta(seconds=30),
}


@dataclass
class TemporalDurability:
    activity_config: workflow.ActivityConfig = field(default_factory=lambda: dict(DEFAULT_ACTIVITY_CONFIG))
    """Base config for every activity."""

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
        unsupported = [
            label
            for label, present in (
                ("team", agent.team),
                ("subagents", agent.subagents),
                ("mcp_servers", agent.mcp_servers),
                ("skills", agent.skills),
                ("memory", agent.memory),
            )
            if present
        ]
        if unsupported:
            raise GwenflowDurabilityError(f"Not supported yet with Temporal: {', '.join(unsupported)}.")
        self._target = agent
        self._activities = self._build_activities()
        # The tokenizer loads its vocabulary lazily (disk cache, network on first use):
        # do it now, outside any workflow, so token counting stays pure in workflow code.
        agent.history.tokenizer_fn("warm-up")

    @property
    def activities(self) -> List[Callable]:
        return list(self._activities)

    def _activity_name(self, kind: str) -> str:
        return f"gwenflow__{self._target.name}__{kind}"

    def _config_for_model(self) -> workflow.ActivityConfig:
        return {**self.activity_config, **self.model_activity_config}

    def _config_for_tool(self, tool_name: str) -> workflow.ActivityConfig:
        return {**self.activity_config, **self.tool_activity_config.get(tool_name, {})}

    # -- worker side: the activities that do the real I/O --------------------

    def _build_activities(self) -> List[Callable]:
        durability = self

        @activity.defn(name=self._activity_name("model_request"))
        async def model_request(request: Dict[str, Any]) -> Dict[str, Any]:
            agent = durability._target
            # One llm copy per call: activities of concurrent workflows share this worker-side agent.
            llm = copy.copy(agent.llm)
            tools_by_name = {t.name: t for t in agent._llm_tools()}
            names = request["tool_names"]
            llm.tools = [tools_by_name[n] for n in names] if names else None
            llm.tool_choice = request["tool_choice"]
            if hasattr(llm, "response_format"):
                llm.response_format = agent.response_model if request["structured_output"] else None
            response = await llm.ainvoke(input=request["messages"])
            return _RESPONSE.dump_python(response, mode="json")

        @activity.defn(name=self._activity_name("call_tool"))
        async def call_tool(request: Dict[str, Any]) -> Optional[str]:
            tool = {t.name: t for t in durability._target.get_all_tools()}[request["name"]]
            # A single attempt: retries belong to Temporal, not to tenacity inside the activity.
            result = await asyncio.to_thread(tool._run, **request["arguments"])
            return tool._cast_response_to_str(result)

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
            record = await asyncio.to_thread(history.compact)
            return {
                "changed": record is not None,
                "messages": [m.to_dict() for m in history.messages],
                "summary": getattr(history, "summary", None),
                "summarized_upto": getattr(history, "_summarized_upto", 0),
            }

        return [model_request, call_tool, compact_history]

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

    def before_model_request(self, agent: Any) -> None:
        """Hook called by the agent loop before each model request, in workflow code."""
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
        clone = agent.clone()
        clone._workflow_copy = True
        clone.tools = [_ActivityTool.wrap(t, self) for t in agent.tools]
        proxy = _ActivityChat.wrap(agent.llm, self)
        proxy.tools = clone._llm_tools() if clone.tools else None
        proxy.tool_choice = agent.llm.tool_choice
        proxy.response_format = getattr(agent.llm, "response_format", None)
        clone.llm = proxy
        # Over-budget compaction (the one that calls the summarizer) becomes an activity.
        clone.history.durable_compactor = self._compact_in_activity
        self._copies[instance] = clone
        return clone

    async def execute_node(self, node: Node, input_data: Optional[Dict[str, Any]]) -> Any:
        """Flow side: run one node of the bound flow as a `gwenflow__flow_node` activity."""
        return await workflow.execute_activity(
            FLOW_NODE_ACTIVITY,
            {"node": asdict(node), "input_data": input_data},
            **node_activity_config(node, self._target),
        )
