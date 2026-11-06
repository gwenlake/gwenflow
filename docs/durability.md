# Durable execution with Temporal

`gwenflow.durability` runs agents and flows on [Temporal](https://temporal.io): a run
survives worker crashes, deployments and restarts, and resumes where it stopped instead of
starting over. Agents and flows stay the same objects; attaching a `TemporalDurability`
makes them durable when they run inside a Temporal workflow, and changes nothing outside
one.

```bash
pip install "gwenflow[temporal]"
temporal server start-dev   # local Temporal server for development
```

`import gwenflow` does not need Temporal; only `import gwenflow.durability` does.

Examples:

- [examples/demo_agent_temporal.py](../examples/demo_agent_temporal.py): a durable
  agent, the simplest setup.
- [examples/agent_temporal.py](../examples/agent_temporal.py): a durable agent whose
  trade-booking tool waits for a human approval.
- [examples/demo_flow_temporal.py](../examples/demo_flow_temporal.py): a YAML flow run
  durably, then scheduled every day.

## How it works

Temporal records every side effect of a workflow in its history. On a crash, another
worker replays the workflow code and reuses the recorded results instead of executing the
side effects again. Workflow code must therefore be deterministic, and all I/O must run in
**activities**.

With a `TemporalDurability` attached:

| Runs in workflow code (deterministic) | Runs as an activity (I/O, retried by Temporal) |
|---|---|
| the agent loop, the history, tool dispatch | each model request: `gwenflow__<agent>__model_request` |
| the DAG walk of a flow | each tool call: `gwenflow__<agent>__call_tool` |
| token counting | compaction that calls the summarizer: `gwenflow__<agent>__compact_history` |
| | each flow node: `gwenflow__flow_node` |

A replayed run does not call the model or the tools again: their results come from the
history.

## Durable agents

```python
from temporalio import workflow
from temporalio.worker import Worker

with workflow.unsafe.imports_passed_through():
    from gwenflow import Agent, ChatOpenAI, Tool
    from gwenflow.durability import TemporalDurability, worker_options

agent = Agent(
    name="finance",                   # stable: activity names derive from it
    llm=ChatOpenAI(model="gpt-4o-mini"),
    tools=[Tool(get_exchange_rate)],
    durability=TemporalDurability(),  # one instance per agent
)


@workflow.defn
class Finance:
    @workflow.run
    async def run(self, prompt: str) -> str:
        first = await agent.arun(prompt)
        second = await agent.arun("And for the yen?")  # same history as the first call
        return second.content


worker = Worker(client, task_queue="finance", **worker_options(Finance, agents=[agent]))
```

- **Module-level agent, per-run copy.** Inside a workflow, `agent.arun()` runs on a copy
  of the agent created for the current workflow run: its own history, its llm and tools
  proxied to activities. Every `arun()` of the run reuses that copy, so a multi-turn
  conversation keeps its history, and a replay rebuilds it identically. The same agent
  object serves every concurrent run.
- **Name.** The agent needs a `name` made of letters, digits, `_` or `-`. It is part of
  the activity names, so do not rename an agent with runs in flight.
- **Async only.** Use `await agent.arun()` in a workflow; `agent.run()` raises
  `GwenflowDurabilityError`.
- **Context window.** `context_window="auto"` queries the model server, which a workflow
  cannot do: give an int (or leave it unset to use the built-in table).
- An agent without durability raises an explicit error when it runs inside a workflow.

### Tool approvals with `AgentWorkflow`

Tools created with `requires_approval=True` pause the run. `AgentWorkflow` is an optional
base class that waits durably for the decision (hours or days if needed) and exposes it
to the outside world:

```python
from gwenflow.durability import AgentWorkflow


@workflow.defn
class FxDesk(AgentWorkflow):
    __gwenflow_agents__ = [agent]   # registered by worker_options(FxDesk)

    @workflow.run
    async def run(self, prompt: str) -> str:
        response = await self.run_agent(agent, prompt, timeout=timedelta(hours=1))
        return response.content
```

From a client:

```python
handle = await client.start_workflow(FxDesk.run, "Convert 1000 USD into euros.", id="fx-1", task_queue="fx")
pending = await handle.query(FxDesk.get_pending_approvals)   # [{"tool_call_id", "tool_name", "tool_args", ...}]
await handle.signal(FxDesk.approve, args=[pending[0]["tool_call_id"], True])  # or False to deny
print(await handle.result())
```

If `timeout` expires before every pending call is decided, `run_agent` raises
`asyncio.TimeoutError`.

### Activity options

```python
from datetime import timedelta
from temporalio.common import RetryPolicy

TemporalDurability(
    activity_config={"start_to_close_timeout": timedelta(seconds=60)},           # every activity
    model_activity_config={"start_to_close_timeout": timedelta(minutes=10)},     # model requests and summarizer
    tool_activity_config={                                                       # per tool name
        "search_web": {"start_to_close_timeout": timedelta(minutes=2), "retry_policy": RetryPolicy(maximum_attempts=3)},
    },
)
```

Each value is a Temporal activity config (`start_to_close_timeout`, `heartbeat_timeout`,
`retry_policy`...). `model_activity_config` and the `tool_activity_config` entries are
merged over `activity_config`. Defaults: 60 s for tools, 5 min with a 30 s heartbeat for
model requests, Temporal's default retry policy. Retries belong to Temporal: a tool
runs a single attempt inside its activity.

### Long conversations: `continue_as_new`

A Temporal history is limited in size and event count. For an agent that runs for a very
long time, pass `continue_as_new`: before each model request, when Temporal suggests it,
it is called with the workflow-side agent and should restart the workflow with what it
needs to resume.

```python
def restart(agent):
    workflow.continue_as_new(args=[[m.to_dict() for m in agent.history.messages]])

agent = Agent(name="assistant", llm=..., durability=TemporalDurability(continue_as_new=restart))


@workflow.defn
class Assistant:
    @workflow.run
    async def run(self, messages: list) -> str:
        # The per-run copy of the agent, seeded with the history carried over.
        copy = agent.durability.workflow_agent(agent)
        for message in messages:
            copy.history.add_message(message)
        ...
```

What to carry over and how to resume is up to the workflow: Gwenflow only calls the hook
at a safe point, between two model requests.

## Durable flows

```python
flow = Flow.from_file("examples/flows/news_summary.yaml", durability=TemporalDurability())


@workflow.defn
class NewsSummary:
    @workflow.run
    async def run(self) -> str:
        results = await flow.arun()   # each node runs as an activity
        return results["summarize"]


worker = Worker(client, task_queue="flows", **worker_options(NewsSummary))
```

- Independent branches run in parallel, as outside Temporal.
- Each node is a `gwenflow__flow_node` activity. Its timeout and retries come from the
  node's `settings` (`timeout` in seconds, `retries`), then from the flow's
  `settings` (`node_timeout`, `node_retries`), then default to 300 s and 2 retries.
  Long nodes heartbeat every 10 s, so a dead worker is detected in 30 s.
- Node `credentials` (parameter name -> environment variable) are resolved on the worker
  when the node runs: secrets never enter the workflow history.
- A node that exhausts its retries raises `FlowNodeError` in the workflow. Catch it, or
  let it fail the run: `worker_options()` declares `FlowError` as a workflow failure, so
  the run fails instead of retrying the workflow task forever.

### `DurableFlow`: a ready-made workflow

`DurableFlow` takes a flow definition as input (`Flow.to_dict()`, or the YAML loaded as a
dict), so a run keeps the version of the flow it started with, whatever happens to the
file afterwards. It returns every node's result, keyed by node name, and reports progress
through a query:

```python
from gwenflow.durability import DurableFlow

worker = Worker(client, task_queue="flows", **worker_options(DurableFlow))

handle = await client.start_workflow(DurableFlow.run, flow.to_dict(), id="news-1", task_queue="flows")
await handle.query(DurableFlow.progress)   # {"fetch_news": "done", "summarize": "running"}
results = await handle.result()
```

An invalid definition fails the run with an `InvalidFlow` application error, a node out
of retries with a `FlowNodeError` one.

### Scheduling

A durable flow is an ordinary workflow, so Temporal schedules run it on a timer; see
[examples/demo_flow_temporal.py](../examples/demo_flow_temporal.py):

```python
await client.create_schedule(
    "news-summary-daily",
    Schedule(
        action=ScheduleActionStartWorkflow(NewsSummary.run, id="news-summary", task_queue="flows"),
        spec=ScheduleSpec(intervals=[ScheduleIntervalSpec(every=timedelta(days=1))]),
    ),
)
```

## The worker: `worker_options()`

```python
Worker(client, task_queue="...", **worker_options(*workflow_classes, agents=[...]))
```

returns the `workflows`, `activities`, `workflow_runner` and
`workflow_failure_exception_types` arguments of a Temporal `Worker`:

- the workflow classes passed in;
- the flow node activity, and the activities of every durable agent passed in `agents`
  or declared in a workflow's `__gwenflow_agents__`;
- a sandbox that lets Gwenflow and the provider SDKs through (imported once, outside of
  it). Import them in workflow modules under `workflow.unsafe.imports_passed_through()`;
- `FlowError` as a workflow failure.

## Large payloads: `ClaimCheckCodec`

Temporal limits a payload to 2 MB. With `ClaimCheckCodec`, any payload above a threshold
(128 KB by default) is stored in a blob store and the history keeps only its content hash:
a 50 MB CSV node or a long agent history no longer hits the limit.

```python
import dataclasses
from temporalio.client import Client
from temporalio.converter import DataConverter
from gwenflow.durability import ClaimCheckCodec, S3BlobStore

store = S3BlobStore("my-bucket", prefix="temporal-payloads/", endpoint_url="https://s3.fr-par.scw.cloud")
converter = dataclasses.replace(DataConverter.default, payload_codec=ClaimCheckCodec(store, threshold=128 * 1024))
client = await Client.connect("localhost:7233", data_converter=converter)
```

`S3BlobStore` works with any S3-compatible bucket (extra keyword arguments go to
`boto3.client`). `InMemoryBlobStore` is for tests and single-process demos. Any object
with async `put(key, data)` and `get(key)` methods can serve as a store. Blobs must live
as long as the histories that reference them can be replayed, and the client and the
workers must use the same codec.

## Not supported yet

A durable agent cannot have a team or sub-agents, MCP servers, skills or long-term
memory: `TemporalDurability` raises `GwenflowDurabilityError` when the agent is built.
Inside a workflow, streaming and the synchronous API (`agent.run()`) raise the same
error.
