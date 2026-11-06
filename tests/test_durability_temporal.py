"""Durable execution with Temporal: durable agents and flows, `AgentWorkflow`, `DurableFlow`.

Runs against Temporal's time-skipping test server (downloaded by the SDK on
first use). LLMs are stubs and tiktoken is replaced by a byte encoder, so no
HTTP call is made to a model provider.

Workflows, agents and node types are defined at module level and nothing runs
at import: the Temporal sandbox re-imports workflow modules.
"""

import pytest

pytest.importorskip("temporalio")

import asyncio  # noqa: E402
import dataclasses  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from datetime import timedelta  # noqa: E402
from pathlib import Path  # noqa: E402

import tiktoken  # noqa: E402
from temporalio import workflow  # noqa: E402
from temporalio.client import Client, WorkflowFailureError  # noqa: E402
from temporalio.converter import DataConverter  # noqa: E402
from temporalio.testing import WorkflowEnvironment  # noqa: E402
from temporalio.worker import Replayer, Worker  # noqa: E402
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner  # noqa: E402

with workflow.unsafe.imports_passed_through():
    from gwenflow import Agent, Tool
    from gwenflow.durability import (
        AgentWorkflow,
        ClaimCheckCodec,
        DurableFlow,
        InMemoryBlobStore,
        TemporalDurability,
        worker_options,
    )
    from gwenflow.flows import Flow, FlowNodeError
    from gwenflow.flows.handlers import node_type
    from gwenflow.llms.base import ChatBase
    from gwenflow.memory import ChatMemoryBuffer
    from gwenflow.types import ModelResponse
    from gwenflow.types.message import TextContent, ToolCall

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Offline tokenizer
# ---------------------------------------------------------------------------


class _ByteEncoding:
    """Stand-in for a tiktoken encoding: one token per UTF-8 byte, no vocabulary download."""

    def encode(self, text, **kw):
        return list(str(text).encode())

    def decode(self, ids, **kw):
        return bytes(ids).decode(errors="ignore")


def _count_bytes(text) -> int:
    return len(str(text).encode())


@pytest.fixture(autouse=True)
def offline_tiktoken(monkeypatch):
    monkeypatch.setattr(tiktoken, "get_encoding", lambda name: _ByteEncoding())
    monkeypatch.setattr(tiktoken, "encoding_for_model", lambda name: _ByteEncoding())


def _history() -> ChatMemoryBuffer:
    # The durable agents are built at import, before any fixture runs, and
    # `TemporalDurability.bind()` warms the tokenizer up: count bytes instead.
    return ChatMemoryBuffer(max_fraction=0.85, tokenizer_fn=_count_bytes)


# ---------------------------------------------------------------------------
# Stub LLMs and tools
# ---------------------------------------------------------------------------

CALLS = {"llm": 0, "tool": 0}


@dataclass(kw_only=True)
class TurnChat(ChatBase):
    """Calls the tool once per user turn, then answers with what it has seen."""

    response_format: object = None

    async def ainvoke(self, input=None, **kw):
        CALLS["llm"] += 1
        last_user = max(i for i, m in enumerate(input) if m.get("role") == "user")
        tool_after = [m for m in input[last_user:] if m.get("role") == "tool"]
        if not tool_after:
            return ModelResponse(parts=[ToolCall(id=f"call_{CALLS['llm']}", name="get_rate", arguments="{}")])
        users = [m["content"] for m in input if m.get("role") == "user"]
        return ModelResponse(
            parts=[TextContent(content=f"turn {len(users)}: {users[-1]} -> {tool_after[-1]['content']}")]
        )

    def invoke(self, *a, **k):
        raise NotImplementedError

    def stream(self, *a, **k):
        raise NotImplementedError

    async def astream(self, *a, **k):
        raise NotImplementedError
        yield


@dataclass(kw_only=True)
class RateChat(ChatBase):
    """Asks for the EUR rate, then reports the tool result."""

    response_format: object = None

    async def ainvoke(self, input=None, **kw):
        CALLS["llm"] += 1
        tool_messages = [m for m in input if m.get("role") == "tool"]
        if not tool_messages:
            return ModelResponse(
                parts=[ToolCall(id="call_1", name="get_currency_rate", arguments='{"currency": "EUR"}')]
            )
        return ModelResponse(parts=[TextContent(content="1 EUR = " + tool_messages[-1]["content"] + " USD")])

    def invoke(self, *a, **k):
        raise NotImplementedError

    def stream(self, *a, **k):
        raise NotImplementedError

    async def astream(self, *a, **k):
        raise NotImplementedError
        yield


def get_rate() -> str:
    """Current EUR/USD rate."""
    CALLS["tool"] += 1
    return "1.08"


def get_currency_rate(currency: str) -> str:
    """Exchange rate for a currency."""
    CALLS["tool"] += 1
    return "1.08"


# ---------------------------------------------------------------------------
# Flow node types
# ---------------------------------------------------------------------------

ATTEMPTS: dict[str, int] = {}
SPANS: dict[str, tuple[float, float]] = {}
_lock = threading.Lock()


@node_type("temporal_test.Value")
def _value(parameters, input_data):
    return parameters["value"]


@node_type("temporal_test.Sleep")
def _sleep(parameters, input_data):
    start = time.monotonic()
    time.sleep(parameters["seconds"])
    SPANS[parameters["value"]] = (start, time.monotonic())
    return parameters["value"]


@node_type("temporal_test.Flaky")
def _flaky(parameters, input_data):
    with _lock:
        ATTEMPTS[parameters["key"]] = ATTEMPTS.get(parameters["key"], 0) + 1
        attempt = ATTEMPTS[parameters["key"]]
    if attempt <= parameters["failures"]:
        raise RuntimeError("transient error")
    return f"ok after {attempt} attempts"


@node_type("temporal_test.Big")
def _big(parameters, input_data):
    return {"token_seen": parameters.get("token") == "s3cret", "rows": ["x" * 100] * 5000}  # ~500 KB


@node_type("temporal_test.Join")
def _join(parameters, input_data):
    return {name: (len(v["rows"]) if isinstance(v, dict) else v) for name, v in sorted(input_data.items())}


# ---------------------------------------------------------------------------
# Module-level agents, flow and workflows (nothing Temporal-specific but the durability)
# ---------------------------------------------------------------------------

analyst = Agent(
    name="analyst",
    llm=TurnChat(model="stub", context_window=128000),
    tools=[Tool(get_rate)],
    history=_history(),
    durability=TemporalDurability(),
)
brief_flow = Flow.from_file(FIXTURES / "brief.yaml", durability=TemporalDurability())

finance = Agent(
    name="finance",
    llm=RateChat(model="gpt-5-mini", context_window=128000),
    tools=[Tool(get_currency_rate)],
    history=_history(),
    durability=TemporalDurability(),
)
finance_approval = Agent(
    name="finance-approval",
    llm=RateChat(model="gpt-5-mini", context_window=128000),
    tools=[Tool(get_currency_rate, requires_approval=True)],
    history=_history(),
    durability=TemporalDurability(),
)


@workflow.defn
class MorningBrief:  # no Gwenflow base class
    @workflow.run
    async def run(self, prompt: str) -> list:
        inputs = await brief_flow.arun()
        first = await analyst.arun(f"{prompt} ({inputs['news']})")
        second = await analyst.arun("And tomorrow?")  # same per-run copy: history kept
        return [inputs, first.content, second.content]


@workflow.defn
class FinanceWorkflow(AgentWorkflow):
    __gwenflow_agents__ = [finance]

    @workflow.run
    async def run(self, prompt: str) -> str:
        return (await self.run_agent(finance, prompt)).content


@workflow.defn
class ApprovalWorkflow(AgentWorkflow):
    __gwenflow_agents__ = [finance_approval]

    @workflow.run
    async def run(self, prompt: str) -> str:
        return (await self.run_agent(finance_approval, prompt)).content


failing_flow = Flow.from_dict(
    {
        "name": "failing",
        "nodes": [
            {
                "name": "down",
                "type": "temporal_test.Flaky",
                "parameters": {"key": "down", "failures": 99},
                "settings": {"retries": 0},
            }
        ],
    },
    durability=TemporalDurability(),
)


@workflow.defn
class FailingFlowWorkflow:
    @workflow.run
    async def run(self) -> dict:
        return await failing_flow.arun()


@workflow.defn
class CatchingFlowWorkflow:
    @workflow.run
    async def run(self) -> str:
        try:
            await failing_flow.arun()
        except FlowNodeError as e:
            return f"caught {e.node}"
        return "not raised"


FLOW = {
    "name": "demo",
    "nodes": [
        {"name": "a", "type": "temporal_test.Sleep", "parameters": {"seconds": 1.0, "value": "A"}},
        {"name": "b", "type": "temporal_test.Sleep", "parameters": {"seconds": 1.0, "value": "B"}},
        {
            "name": "flaky",
            "type": "temporal_test.Flaky",
            "parameters": {"key": "k", "failures": 2},
            "settings": {"retries": 3},
        },
        {"name": "big", "type": "temporal_test.Big", "credentials": {"token": "DEMO_TOKEN"}},
        {"name": "join", "type": "temporal_test.Join"},
    ],
    "connections": {n: [[{"node": "join"}]] for n in ("a", "b", "flaky", "big")},
}

BROKEN = {
    "name": "broken",
    "nodes": [
        {
            "name": "never",
            "type": "temporal_test.Flaky",
            "parameters": {"key": "n", "failures": 99},
            "settings": {"retries": 1},
        }
    ],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _options(*workflow_classes, **kwargs):
    """`worker_options()`, with this module passed through the sandbox.

    Users' workflow modules are re-imported in the sandbox; here the test
    module is shared instead, so the counters checked below (and the node
    types registered at import) are the ones the activities update.
    """
    opts = worker_options(*workflow_classes, **kwargs)
    restrictions = opts["workflow_runner"].restrictions.with_passthrough_modules(__name__)
    opts["workflow_runner"] = SandboxedWorkflowRunner(restrictions=restrictions)
    return opts


async def _payload_bytes(history, converter) -> bytes:
    """Every payload recorded in `history`, decoded (claim checks resolved), concatenated."""
    fields = [
        ("workflow_execution_started_event_attributes", "input"),
        ("workflow_execution_completed_event_attributes", "result"),
        ("activity_task_scheduled_event_attributes", "input"),
        ("activity_task_completed_event_attributes", "result"),
    ]
    payloads = []
    for event in history.events:
        for attributes, name in fields:
            if event.HasField(attributes):
                payloads.extend(getattr(getattr(event, attributes), name).payloads)
    decoded = await converter.payload_codec.decode(payloads)
    return b"".join(p.data for p in decoded)


def _activity_types(history) -> list[str]:
    return [
        e.activity_task_scheduled_event_attributes.activity_type.name
        for e in history.events
        if e.HasField("activity_task_scheduled_event_attributes")
    ]


@pytest.fixture(autouse=True)
def reset_state():
    CALLS.update(llm=0, tool=0)
    ATTEMPTS.clear()
    SPANS.clear()
    yield
    for agent in (analyst, finance, finance_approval):
        agent.history.reset()


# ---------------------------------------------------------------------------
# A durable agent and flow in an ordinary workflow
# ---------------------------------------------------------------------------


def test_durable_objects_run_normally_outside_a_workflow():
    async def go():
        return await brief_flow.arun(), await analyst.arun("Local check")

    flow_results, response = asyncio.run(go())

    assert flow_results == {"news": "rates are stable", "markets": "EUR/USD 1.08"}
    assert response.content == "turn 1: Local check -> 1.08"
    assert CALLS == {"llm": 2, "tool": 1}


def test_durable_agent_and_flow_in_a_workflow():
    async def go():
        async with await WorkflowEnvironment.start_time_skipping() as env:
            opts = _options(MorningBrief, agents=[analyst])
            async with Worker(env.client, task_queue="briefs", **opts):
                out = await env.client.execute_workflow(MorningBrief.run, "Brief me", id="brief-1", task_queue="briefs")
                history = await env.client.get_workflow_handle("brief-1").fetch_history()
                llm_calls = CALLS["llm"]
                await Replayer(workflows=opts["workflows"], workflow_runner=opts["workflow_runner"]).replay_workflow(
                    history
                )
                replay_llm_calls = CALLS["llm"] - llm_calls
                out2 = await env.client.execute_workflow(
                    MorningBrief.run, "Other run", id="brief-2", task_queue="briefs"
                )
                return out, history, replay_llm_calls, out2

    out, history, replay_llm_calls, out2 = asyncio.run(go())

    inputs, first, second = out
    assert inputs == {"news": "rates are stable", "markets": "EUR/USD 1.08"}
    assert first == "turn 1: Brief me (rates are stable) -> 1.08"
    # The second turn sees the first one: the per-run copy keeps its history.
    assert second == "turn 2: And tomorrow? -> 1.08"

    model, tool = "gwenflow__analyst__model_request", "gwenflow__analyst__call_tool"
    assert _activity_types(history) == [
        "gwenflow__flow_node",
        "gwenflow__flow_node",
        model,
        tool,
        model,
        model,
        tool,
        model,
    ]
    assert replay_llm_calls == 0  # replay reads results from history, no model call

    # Another run starts from an empty history.
    assert out2[1] == "turn 1: Other run (rates are stable) -> 1.08"
    assert out2[2] == "turn 2: And tomorrow? -> 1.08"
    # The module-level agent itself is untouched by the workflow runs.
    assert analyst.history.messages == []


# ---------------------------------------------------------------------------
# AgentWorkflow
# ---------------------------------------------------------------------------


def test_agent_workflow_runs_and_replays():
    async def go():
        async with await WorkflowEnvironment.start_time_skipping() as env:
            opts = _options(FinanceWorkflow)
            async with Worker(env.client, task_queue="q", **opts):
                out = await env.client.execute_workflow(
                    FinanceWorkflow.run,
                    "Taux EUR ?",
                    id="wf-1",
                    task_queue="q",
                    execution_timeout=timedelta(seconds=40),
                )
                history = await env.client.get_workflow_handle("wf-1").fetch_history()
                calls = dict(CALLS)
                await Replayer(workflows=opts["workflows"], workflow_runner=opts["workflow_runner"]).replay_workflow(
                    history
                )
                return out, history, calls

    out, history, calls = asyncio.run(go())

    assert out == "1 EUR = 1.08 USD"
    assert calls == {"llm": 2, "tool": 1}
    assert _activity_types(history) == [
        "gwenflow__finance__model_request",
        "gwenflow__finance__call_tool",
        "gwenflow__finance__model_request",
    ]
    assert CALLS == calls  # deterministic replay: no new model or tool call


def test_agent_workflow_tool_approval_by_query_and_signal():
    async def go():
        async with await WorkflowEnvironment.start_time_skipping() as env:
            async with Worker(env.client, task_queue="q2", **_options(ApprovalWorkflow)):
                handle = await env.client.start_workflow(ApprovalWorkflow.run, "Taux EUR ?", id="wf-2", task_queue="q2")
                pending = []
                for _ in range(100):
                    pending = await handle.query(ApprovalWorkflow.get_pending_approvals)
                    if pending:
                        break
                    await asyncio.sleep(0.1)
                tool_calls_before_approval = CALLS["tool"]
                await handle.signal(ApprovalWorkflow.approve, args=[pending[0]["tool_call_id"], True])
                return pending, tool_calls_before_approval, await handle.result()

    pending, tool_calls_before_approval, result = asyncio.run(go())

    assert len(pending) == 1
    assert pending[0]["tool_name"] == "get_currency_rate"
    assert tool_calls_before_approval == 0
    assert result == "1 EUR = 1.08 USD"
    assert CALLS["tool"] == 1


# ---------------------------------------------------------------------------
# DurableFlow
# ---------------------------------------------------------------------------


def test_durable_flow(monkeypatch):
    monkeypatch.setenv("DEMO_TOKEN", "s3cret")
    store = InMemoryBlobStore()
    converter = dataclasses.replace(DataConverter.default, payload_codec=ClaimCheckCodec(store))

    async def go():
        async with await WorkflowEnvironment.start_time_skipping() as env:
            client = await Client.connect(env.client.service_client.config.target_host, data_converter=converter)
            opts = _options(DurableFlow)
            async with Worker(client, task_queue="flows", **opts):
                handle = await client.start_workflow(DurableFlow.run, FLOW, id="flow-1", task_queue="flows")
                progress = await handle.query(DurableFlow.progress)
                result = await handle.result()
                history = await handle.fetch_history()
                await Replayer(
                    workflows=opts["workflows"], workflow_runner=opts["workflow_runner"], data_converter=converter
                ).replay_workflow(history)
                recorded = await _payload_bytes(history, converter)

                with pytest.raises(WorkflowFailureError) as failure:
                    await client.execute_workflow(DurableFlow.run, BROKEN, id="flow-2", task_queue="flows")
                return progress, result, history, recorded, failure.value

    progress, result, history, recorded, failure = asyncio.run(go())

    assert set(progress) == {"a", "b", "flaky", "big", "join"}
    assert result["join"] == {"a": "A", "b": "B", "big": 5000, "flaky": "ok after 3 attempts"}
    assert result["big"]["token_seen"] is True

    # Independent branches run in parallel.
    (a_start, a_end), (b_start, b_end) = SPANS["A"], SPANS["B"]
    assert a_start < b_end and b_start < a_end

    # Per-node retries: two failures, then success on the third attempt.
    assert ATTEMPTS["k"] == 3

    # Credentials are resolved on the worker, never recorded in history.
    assert b"DEMO_TOKEN" in recorded  # the variable name is recorded...
    assert b"s3cret" not in recorded  # ...never its value

    # Claim check: the ~500 KB payloads are offloaded, history events stay small.
    assert store.blobs
    assert max(e.ByteSize() for e in history.events) < 128 * 1024

    # A node that exhausts its attempts fails the run cleanly.
    assert ATTEMPTS["n"] == 2  # 1 + retries
    assert "Node 'never' failed" in failure.cause.message
    assert failure.cause.type == "FlowNodeError"


# ---------------------------------------------------------------------------
# A failing flow in a custom workflow
# ---------------------------------------------------------------------------


def test_failing_flow_fails_a_custom_workflow_run():
    """A node out of retries fails the run (no endless workflow task retries); workflow code can catch it."""

    async def go():
        async with await WorkflowEnvironment.start_time_skipping() as env:
            async with Worker(env.client, task_queue="custom", **_options(FailingFlowWorkflow, CatchingFlowWorkflow)):
                with pytest.raises(WorkflowFailureError) as failure:
                    await asyncio.wait_for(
                        env.client.execute_workflow(FailingFlowWorkflow.run, id="custom-1", task_queue="custom"),
                        timeout=30,
                    )
                caught = await asyncio.wait_for(
                    env.client.execute_workflow(CatchingFlowWorkflow.run, id="custom-2", task_queue="custom"),
                    timeout=30,
                )
                return failure.value, caught

    failure, caught = asyncio.run(go())

    assert failure.cause.type == "FlowNodeError"
    assert "Node 'down' failed" in failure.cause.message
    assert caught == "caught down"
