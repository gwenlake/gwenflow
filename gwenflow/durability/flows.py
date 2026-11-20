"""Flow side of `TemporalDurability`: the node activity, and `DurableFlow`, a ready-made workflow."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Any, Callable, Dict

from temporalio import activity, workflow
from temporalio.exceptions import ApplicationError

from gwenflow.flows.flows import Flow, FlowError, FlowNodeError, Node, call_handler, resolve_credentials

FLOW_NODE_ACTIVITY = "gwenflow__flow_node"
DEFAULT_NODE_TIMEOUT = 300.0
DEFAULT_NODE_RETRIES = 2
HEARTBEAT_EVERY = 10.0


@activity.defn(name=FLOW_NODE_ACTIVITY)
async def flow_node(request: Dict[str, Any]) -> Any:
    """Run one flow node. Credentials are resolved here, on the worker, never in history."""
    node = Node(**request["node"])
    try:
        parameters = resolve_credentials(node)
    except FlowNodeError as e:
        raise ApplicationError(str(e), type="MissingCredential", non_retryable=True) from e
    if node.type not in _node_types():
        raise ApplicationError(f"Node type '{node.type}' is not supported.", type="UnknownNodeType", non_retryable=True)
    task = asyncio.ensure_future(asyncio.to_thread(call_handler, node.type, parameters, request["input_data"]))
    while not task.done():  # long nodes (agents, big queries) keep reporting they are alive
        activity.heartbeat()
        await asyncio.wait({task}, timeout=HEARTBEAT_EVERY)
    return task.result()


def _node_types() -> Dict[str, Callable]:
    from gwenflow.flows.handlers import NODE_TYPE_REGISTRY

    return NODE_TYPE_REGISTRY


def node_activity_config(node: Node, flow: Flow) -> workflow.ActivityConfig:
    from temporalio.common import RetryPolicy

    timeout = node.settings.get("timeout", flow.settings.get("node_timeout", DEFAULT_NODE_TIMEOUT))
    retries = node.settings.get("retries", flow.settings.get("node_retries", DEFAULT_NODE_RETRIES))
    return {
        "start_to_close_timeout": timedelta(seconds=float(timeout)),
        "heartbeat_timeout": timedelta(seconds=3 * HEARTBEAT_EVERY),
        "retry_policy": RetryPolicy(
            maximum_attempts=1 + int(retries),
            initial_interval=timedelta(seconds=1),
            maximum_interval=timedelta(seconds=30),
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
        """Node name -> pending | running | done | failed."""
        return self.status

    @workflow.run
    async def run(self, definition: Dict[str, Any]) -> Dict[str, Any]:
        from gwenflow.durability.temporal import TemporalDurability

        try:
            flow = Flow.from_dict(definition, durability=TemporalDurability())
        except FlowError as e:
            raise ApplicationError(str(e), type="InvalidFlow", non_retryable=True) from e
        self.status = dict.fromkeys(flow.topological_order(), "pending")
        try:
            return await flow.arun(on_status=self.status.__setitem__)
        except FlowNodeError as e:
            # A node that exhausted its retries fails the run, not the workflow task.
            raise ApplicationError(str(e), {"node": e.node}, type="FlowNodeError", non_retryable=True) from e
