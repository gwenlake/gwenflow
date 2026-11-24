"""Flow side of `TemporalDurability`: the node activity, and `DurableFlow`, a ready-made workflow."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Any, Callable, Dict, Optional

from temporalio import activity, workflow
from temporalio.exceptions import ApplicationError

from gwenflow.flows.flows import (
    NON_RETRYABLE_ERRORS,
    RETRY_INITIAL_INTERVAL,
    RETRY_MAX_INTERVAL,
    Flow,
    FlowError,
    FlowNodeError,
    Node,
    call_handler,
    node_execution,
)
from gwenflow.flows.policy import FlowPolicy, FlowPolicyError
from gwenflow.flows.runtime import NodeControl, start_in_thread

FLOW_NODE_ACTIVITY = "gwenflow__flow_node"
MAX_NODE_DURATION = 3600.0
HEARTBEAT_EVERY = 10.0


class FlowNodeActivity:
    def __init__(self, policy: Optional[FlowPolicy] = None) -> None:
        self.policy = policy or FlowPolicy()

    @activity.defn(name=FLOW_NODE_ACTIVITY)
    async def run(self, request: Dict[str, Any]) -> Any:
        """Run one flow node. Credentials are resolved here, on the worker, never in history."""
        node = Node(**request["node"])
        try:
            parameters = self.policy.resolve(node)
        except FlowPolicyError as e:
            raise ApplicationError(e.reason, type="FlowPolicyViolation", non_retryable=True) from e
        except FlowNodeError as e:
            raise ApplicationError(e.reason, type="MissingCredential", non_retryable=True) from e
        if node.type not in _node_types():
            raise ApplicationError(
                f"Node type '{node.type}' is not supported.", type="UnknownNodeType", non_retryable=True
            )
        timeout = activity.info().start_to_close_timeout
        control = NodeControl(timeout.total_seconds() if timeout else None)
        task = start_in_thread(control, call_handler, node.type, parameters, request["input_data"])
        try:
            while not task.done():  # long nodes (agents, big queries) keep reporting they are alive
                activity.heartbeat()
                await asyncio.wait({task}, timeout=HEARTBEAT_EVERY)
        except asyncio.CancelledError:
            control.stop()
            raise
        try:
            return task.result()
        except NON_RETRYABLE_ERRORS as e:
            raise ApplicationError(str(e), type=type(e).__name__, non_retryable=True) from e


flow_node = FlowNodeActivity().run


def _node_types() -> Dict[str, Callable]:
    from gwenflow.flows.handlers import NODE_TYPE_REGISTRY

    return NODE_TYPE_REGISTRY


def node_activity_config(node: Node, flow: Flow) -> workflow.ActivityConfig:
    from temporalio.common import RetryPolicy

    execution = node_execution(node, flow)
    return {
        "start_to_close_timeout": timedelta(seconds=execution.timeout or MAX_NODE_DURATION),
        "heartbeat_timeout": timedelta(seconds=3 * HEARTBEAT_EVERY),
        "retry_policy": RetryPolicy(
            maximum_attempts=execution.attempts,
            initial_interval=timedelta(seconds=RETRY_INITIAL_INTERVAL),
            maximum_interval=timedelta(seconds=RETRY_MAX_INTERVAL),
        ),
    }


@workflow.defn
class DurableFlow:
    """Durable run of a flow definition (`Flow.to_dict()`, or the YAML loaded as a dict).

    The definition is the workflow input, so it is recorded in the history: a
    run keeps the version of the flow it started with, whatever happens to the
    file afterwards. Returns every node's result, keyed by node name.
    """

    def __init__(self) -> None:
        self.status: Dict[str, str] = {}

    @workflow.query
    def progress(self) -> Dict[str, str]:
        """Node name -> pending | running | done | failed | cancelled | skipped."""
        return self.status

    @workflow.run
    async def run(self, definition: Dict[str, Any]) -> Dict[str, Any]:
        from gwenflow.durability.temporal import TemporalDurability

        try:
            flow = Flow.from_dict(definition, durability=TemporalDurability())
            flow.validate()
        except FlowError as e:
            raise ApplicationError(str(e), type="InvalidFlow", non_retryable=True) from e
        self.status = dict.fromkeys(flow.topological_order(), "pending")
        try:
            return await flow.arun(on_status=self.status.__setitem__)
        except FlowNodeError as e:
            # A node that exhausted its retries fails the run, not the workflow task.
            raise ApplicationError(str(e), {"node": e.node}, type="FlowNodeError", non_retryable=True) from e
