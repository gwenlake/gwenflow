"""Agent side of `TemporalDurability`: activity proxies for the model and the tools, and `AgentWorkflow`."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional

from pydantic import TypeAdapter
from temporalio import workflow

from gwenflow.durability.errors import GwenflowDurabilityError
from gwenflow.llms.base import ChatBase
from gwenflow.tools import BaseTool
from gwenflow.types import ModelResponse

if TYPE_CHECKING:
    from gwenflow.durability.temporal import TemporalDurability

_RESPONSE = TypeAdapter(ModelResponse)


@dataclass(kw_only=True)
class _ActivityChat(ChatBase):
    """ChatBase whose requests run as the agent's `model_request` activity."""

    target: ChatBase = None
    durability: TemporalDurability = None
    response_format: Any = None

    @classmethod
    def wrap(cls, target: ChatBase, durability: TemporalDurability) -> "_ActivityChat":
        if target.context_window == "auto":
            raise GwenflowDurabilityError("context_window='auto' queries the server: set an int for durable agents.")
        return cls(
            model=target.model,
            provider=target.provider,
            context_window=target.context_window,
            fallback_context_window=target.fallback_context_window,
            target=target,
            durability=durability,
        )

    async def ainvoke(self, input: Any = None, **kwargs: Any) -> ModelResponse:
        request = {
            "messages": json.loads(json.dumps(input, default=_to_jsonable)),
            "tool_names": [t.name for t in (self.tools or [])],
            "tool_choice": self.tool_choice,
            "structured_output": self.response_format is not None,
        }
        data = await workflow.execute_activity(
            self.durability._activity_name("model_request"), request, **self.durability._config_for_model()
        )
        return _RESPONSE.validate_python(data)

    def invoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
        raise GwenflowDurabilityError(
            "Synchronous model calls are not possible in a workflow: use `await agent.arun()`."
        )

    def stream(self, *args: Any, **kwargs: Any):
        raise GwenflowDurabilityError("Streaming inside a workflow is not supported yet.")

    async def astream(self, *args: Any, **kwargs: Any):
        raise GwenflowDurabilityError("Streaming inside a workflow is not supported yet.")
        yield  # pragma: no cover

    def get_thinking_parts(self, response: ModelResponse):
        return self.target.get_thinking_parts(response)


@dataclass(kw_only=True)
class _ActivityTool(BaseTool):
    """Same schema as the wrapped tool; `arun` executes the `call_tool` activity."""

    durability: TemporalDurability = None

    @classmethod
    def wrap(cls, target: BaseTool, durability: TemporalDurability) -> "_ActivityTool":
        return cls(
            name=target.name,
            description=target.description,
            parameters=target.parameters,
            function_schema=target.function_schema,
            requires_approval=target.requires_approval,
            durability=durability,
        )

    def __post_init__(self) -> None:  # schema copied from the target, no introspection
        pass

    def _run(self, **kwargs: Any) -> Any:
        raise GwenflowDurabilityError(f"Tool {self.name!r} must be awaited (`arun`) inside a workflow.")

    async def arun(self, **kwargs: Any) -> Optional[str]:
        return await workflow.execute_activity(
            self.durability._activity_name("call_tool"),
            {"name": self.name, "arguments": kwargs},
            **self.durability._config_for_tool(self.name),
        )


def _to_jsonable(obj: Any) -> Any:
    return obj.to_dict() if hasattr(obj, "to_dict") else str(obj)


class AgentWorkflow:
    """Optional base: declares the agents and handles tool approvals."""

    __gwenflow_agents__: ClassVar[List[Any]] = []

    def __init__(self) -> None:
        self._decisions: Dict[str, bool] = {}
        self.pending_approvals: List[Dict[str, Any]] = []

    @workflow.signal
    def approve(self, tool_call_id: str, approved: bool) -> None:
        self._decisions[tool_call_id] = approved

    @workflow.query
    def get_pending_approvals(self) -> List[Dict[str, Any]]:
        return self.pending_approvals

    async def run_agent(self, agent: Any, input: Any, timeout: Optional[timedelta] = None) -> Any:
        """Run `agent` (its per-workflow copy, see `TemporalDurability`), waiting durably on each approval."""
        response = await agent.arun(input)
        while response.pending_approvals:
            self.pending_approvals = [a.to_dict() for a in response.pending_approvals]
            ids = [a["tool_call_id"] for a in self.pending_approvals]
            await workflow.wait_condition(lambda ids=ids: all(i in self._decisions for i in ids), timeout=timeout)
            self.pending_approvals = []
            response = await agent.arun(None, approvals={i: self._decisions[i] for i in ids})
        return response
