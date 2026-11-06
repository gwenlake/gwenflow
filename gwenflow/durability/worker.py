"""Worker wiring for Gwenflow workflows."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List

from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner, SandboxRestrictions

from gwenflow.durability.flows import flow_node
from gwenflow.flows.flows import FlowError


def worker_options(*workflow_classes: type, agents: Iterable[Any] = ()) -> Dict[str, Any]:
    """`Worker(client, task_queue=..., **worker_options(MyWorkflow, agents=[agent]))`.

    Registers the workflows, the flow node activity, and every activity of the
    durable agents passed in `agents` or declared in a workflow's
    `__gwenflow_agents__`. Lets Gwenflow and the provider SDKs through the
    sandbox (imported once, outside of it).

    A `FlowError` raised by workflow code (a node that exhausted its retries,
    an invalid flow) fails the workflow run instead of retrying the workflow
    task forever; workflow code can still catch it.
    """
    activities: List[Callable] = [flow_node]
    declared = [a for cls in workflow_classes for a in getattr(cls, "__gwenflow_agents__", [])]
    seen: set[int] = set()
    for agent in [*agents, *declared]:
        if id(agent) in seen:
            continue
        seen.add(id(agent))
        if agent.durability is None:
            raise ValueError(f"Agent '{agent.name}' has no durability: nothing to register.")
        activities.extend(agent.durability.activities)
    restrictions = SandboxRestrictions.default.with_passthrough_modules(
        "gwenflow", "openai", "anthropic", "mistralai", "google", "httpx", "tiktoken", "pydantic", "tenacity", "rich"
    )
    return {
        "workflows": list(workflow_classes),
        "activities": activities,
        "workflow_runner": SandboxedWorkflowRunner(restrictions=restrictions),
        "workflow_failure_exception_types": [FlowError],
    }
