"""Durable execution of Gwenflow agents and flows with Temporal.

Same principle as `pydantic_ai.durable_exec.temporal`: agents and flows stay
the same; attaching `TemporalDurability` makes them durable when they run
inside a Temporal workflow, and changes nothing outside one.

    agent = Agent(name="finance", llm=..., tools=[...], durability=TemporalDurability())
    flow = Flow.from_file("news.yaml", durability=TemporalDurability())

    @workflow.defn
    class MorningBrief:
        @workflow.run
        async def run(self, prompt: str) -> str:
            news = await flow.arun()                   # each node is an activity
            return (await agent.arun(prompt)).content  # model and tools too

    worker = Worker(client, task_queue="briefs", **worker_options(MorningBrief, agents=[agent]))

Inside a workflow, each run gets its own copy of the agent (its own history,
rebuilt identically on replay), so the same module-level agent serves every
run. `AgentWorkflow` (tool approvals) and `DurableFlow` (a flow definition as
input, node progress as a query) are optional ready-made workflows.

Not covered yet: team / subagents (thread based), MCP servers, skills and
long-term memory toolsets, streaming.
"""

try:
    import temporalio  # noqa: F401
except ImportError as e:  # pragma: no cover
    raise ImportError("Durable execution needs Temporal: pip install 'gwenflow[temporal]'") from e

from gwenflow.durability.agents import AgentWorkflow
from gwenflow.durability.claim_check import BlobStore, ClaimCheckCodec, InMemoryBlobStore, S3BlobStore
from gwenflow.durability.errors import GwenflowDurabilityError
from gwenflow.durability.flows import DurableFlow, flow_node
from gwenflow.durability.temporal import TemporalDurability
from gwenflow.durability.worker import worker_options

__all__ = [
    "AgentWorkflow",
    "BlobStore",
    "ClaimCheckCodec",
    "DurableFlow",
    "GwenflowDurabilityError",
    "InMemoryBlobStore",
    "S3BlobStore",
    "TemporalDurability",
    "flow_node",
    "worker_options",
]
