"""Durable execution with Temporal: durable agents and flows, `AgentWorkflow`, `DurableFlow`.

Workflow tests need a Temporal server: there is no in-process mode, the server
holds the history that workflows are replayed from. `_env()` picks one:

- `TEMPORAL_TEST_SERVER_PATH`: an existing time-skipping test server binary;
- `TEMPORAL_CLI_PATH`: an existing `temporal` CLI, run as a local dev server;
- otherwise the time-skipping test server, downloaded by the SDK on first use.

When none is available (offline, blocked download), those tests are skipped
with the reason, except in CI (`CI` set), where they fail; tests marked "no server" below always run. LLMs are stubs and
tiktoken is replaced by a byte encoder, so no HTTP call is made to a model
provider.

Workflows, agents and node types are defined at module level and nothing runs
at import: the Temporal sandbox re-imports workflow modules.
"""

import pytest

pytest.importorskip("temporalio")

import asyncio  # noqa: E402
import dataclasses  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from datetime import timedelta  # noqa: E402
from http.server import BaseHTTPRequestHandler, HTTPServer  # noqa: E402
from pathlib import Path  # noqa: E402

import httpx  # noqa: E402
import tiktoken  # noqa: E402
from temporalio import workflow  # noqa: E402
from temporalio.client import Client, WorkflowFailureError, WorkflowHistory  # noqa: E402
from temporalio.common import RetryPolicy  # noqa: E402
from temporalio.converter import DataConverter  # noqa: E402
from temporalio.exceptions import ApplicationError  # noqa: E402
from temporalio.testing import WorkflowEnvironment  # noqa: E402
from temporalio.worker import Replayer, Worker  # noqa: E402
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner  # noqa: E402

with workflow.unsafe.imports_passed_through():
    from gwenflow import Agent, ChatOpenAI, Tool, ToolReturn
    from gwenflow.durability import (
        AgentWorkflow,
        ClaimCheckCodec,
        DurableFlow,
        GwenflowDurabilityError,
        InMemoryBlobStore,
        TemporalDurability,
        worker_options,
    )
    from gwenflow.durability.temporal import _non_retryable_status
    from gwenflow.flows import Flow, FlowNodeError
    from gwenflow.flows.handlers import node_type
    from gwenflow.llms.base import ChatBase
    from gwenflow.memory import ChatMemoryBuffer
    from gwenflow.types import TOOL_METADATA_KEY, ModelResponse
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


def get_cited_rate(currency: str) -> ToolReturn:
    """Exchange rate for a currency, with its source."""
    CALLS["tool"] += 1
    return ToolReturn(return_value="1.08", metadata={"source": "ECB", "currency": currency})


def get_broken_rate(currency: str) -> str:
    """Exchange rate for a currency (its provider is down)."""
    CALLS["tool"] += 1
    raise ValueError(f"no rate for {currency}")


class ProviderError(Exception):
    """Stand-in for a provider SDK error (openai, anthropic and mistral expose `status_code`)."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


@dataclass(kw_only=True)
class PlanChat(ChatBase):
    """A reasoning model: answers with a plan."""

    response_format: object = None

    async def ainvoke(self, input=None, **kw):
        CALLS["reasoning"] = CALLS.get("reasoning", 0) + 1
        return ModelResponse(parts=[TextContent(content="look the rate up, then answer")])

    def invoke(self, *a, **k):
        raise NotImplementedError

    def stream(self, *a, **k):
        raise NotImplementedError

    async def astream(self, *a, **k):
        raise NotImplementedError
        yield


@dataclass(kw_only=True)
class RejectingChat(ChatBase):
    """Every request fails with the given HTTP status."""

    response_format: object = None
    status_code: int = 400
    wrapped: bool = False
    """As the provider adapters do: the SDK error is the cause of a `RuntimeError`."""

    async def ainvoke(self, input=None, **kw):
        CALLS["llm"] += 1
        if self.wrapped:
            try:
                raise ProviderError(self.status_code)
            except ProviderError as e:
                raise RuntimeError(f"Error in calling provider API: {e}") from e
        raise ProviderError(self.status_code)

    def invoke(self, *a, **k):
        raise NotImplementedError

    def stream(self, *a, **k):
        raise NotImplementedError

    async def astream(self, *a, **k):
        raise NotImplementedError
        yield


@dataclass(kw_only=True)
class AdapterChat(ChatBase):
    """Fails as a provider adapter does: `error`, raised by the SDK, is the cause of a `RuntimeError`."""

    response_format: object = None
    error: object = None

    async def ainvoke(self, input=None, **kw):
        CALLS["llm"] += 1
        try:
            raise self.error
        except Exception as e:
            raise RuntimeError(f"Error in calling provider API: {e}") from e

    def invoke(self, *a, **k):
        raise NotImplementedError

    def stream(self, *a, **k):
        raise NotImplementedError

    async def astream(self, *a, **k):
        raise NotImplementedError
        yield


@dataclass(kw_only=True)
class SlowChat(ChatBase):
    """Answers after `seconds`, longer than the heartbeat timeout of its agent."""

    response_format: object = None
    seconds: float = 3.0

    async def ainvoke(self, input=None, **kw):
        CALLS["llm"] += 1
        await asyncio.sleep(self.seconds)
        return ModelResponse(parts=[TextContent(content="took my time")])

    def invoke(self, *a, **k):
        raise NotImplementedError

    def stream(self, *a, **k):
        raise NotImplementedError

    async def astream(self, *a, **k):
        raise NotImplementedError
        yield


class _Unauthorized(BaseHTTPRequestHandler):
    """A provider endpoint that refuses every request with a 401."""

    def do_POST(self):  # noqa: N802 (name set by `BaseHTTPRequestHandler`)
        CALLS["http"] = CALLS.get("http", 0) + 1
        body = json.dumps({"error": {"message": "bad key", "type": "invalid_request_error", "code": "invalid_api_key"}})
        self.send_response(401)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body.encode())

    def log_message(self, *a):
        pass


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

cited = Agent(
    name="cited",
    llm=RateChat(model="gpt-5-mini", context_window=128000),
    tools=[Tool(get_cited_rate, name="get_currency_rate")],
    history=_history(),
    durability=TemporalDurability(),
)

broken = Agent(
    name="broken",
    llm=RateChat(model="gpt-5-mini", context_window=128000),
    tools=[Tool(get_broken_rate, name="get_currency_rate", max_retries=2)],
    history=_history(),
    durability=TemporalDurability(),
)
thinker = Agent(
    name="thinker",
    llm=RateChat(model="gpt-5-mini", context_window=128000),
    reasoning_model=PlanChat(model="gpt-5-mini", context_window=128000),
    tools=[Tool(get_currency_rate)],
    history=_history(),
    durability=TemporalDurability(),
)
# Options a workflow cannot run yet: the agents are still built, and still usable outside one.
lead = Agent(
    name="lead",
    llm=RateChat(model="gpt-5-mini", context_window=128000),
    team=[Agent(name="helper", llm=RateChat(model="gpt-5-mini", context_window=128000), history=_history())],
    history=_history(),
    durability=TemporalDurability(),
)
auto_window = Agent(
    name="auto-window",
    llm=RateChat(model="gpt-5-mini", context_window="auto", fallback_context_window=128000),
    history=_history(),
    durability=TemporalDurability(),
)
rejected = Agent(
    name="rejected",
    llm=RejectingChat(model="gpt-5-mini", context_window=128000, status_code=400),
    history=_history(),
    durability=TemporalDurability(),
)
throttled = Agent(
    name="throttled",
    llm=RejectingChat(model="gpt-5-mini", context_window=128000, status_code=429),
    history=_history(),
    durability=TemporalDurability(),
)
wrapped_rejected = Agent(
    name="wrapped-rejected",
    llm=RejectingChat(model="gpt-5-mini", context_window=128000, status_code=401, wrapped=True),
    history=_history(),
    durability=TemporalDurability(),
)
wrapped_throttled = Agent(
    name="wrapped-throttled",
    llm=RejectingChat(model="gpt-5-mini", context_window=128000, status_code=429, wrapped=True),
    history=_history(),
    durability=TemporalDurability(),
)
# A request three times longer than the heartbeat timeout: the activity must report it is alive.
patient = Agent(
    name="patient",
    llm=SlowChat(model="gpt-5-mini", context_window=128000, seconds=3.0),
    history=_history(),
    durability=TemporalDurability(
        model_activity_config={
            "start_to_close_timeout": timedelta(seconds=30),
            "heartbeat_timeout": timedelta(seconds=1),
            "retry_policy": RetryPolicy(maximum_attempts=1),
        }
    ),
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


@workflow.defn
class CitedWorkflow(AgentWorkflow):
    __gwenflow_agents__ = [cited]

    @workflow.run
    async def run(self, prompt: str) -> list:
        response = await self.run_agent(cited, prompt)
        tool_message = next(m for m in response.messages if m.role == "tool")
        return [response.content, tool_message.content, tool_message.extra.get(TOOL_METADATA_KEY)]


@workflow.defn
class BrokenToolWorkflow(AgentWorkflow):
    __gwenflow_agents__ = [broken]

    @workflow.run
    async def run(self, prompt: str) -> str:
        return (await self.run_agent(broken, prompt)).content


@workflow.defn
class ThinkerWorkflow(AgentWorkflow):
    __gwenflow_agents__ = [thinker]

    @workflow.run
    async def run(self, prompt: str) -> str:
        return (await self.run_agent(thinker, prompt)).content


@workflow.defn
class LeadWorkflow:  # the agent is not declared: `worker_options()` cannot check it
    @workflow.run
    async def run(self, prompt: str) -> str:
        return (await lead.arun(prompt)).content


@workflow.defn
class RejectedWorkflow(AgentWorkflow):
    __gwenflow_agents__ = [rejected]

    @workflow.run
    async def run(self, prompt: str) -> str:
        return (await self.run_agent(rejected, prompt)).content


@workflow.defn
class PatientWorkflow(AgentWorkflow):
    __gwenflow_agents__ = [patient]

    @workflow.run
    async def run(self, prompt: str) -> str:
        return (await self.run_agent(patient, prompt)).content


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


async def _env() -> WorkflowEnvironment:
    """A Temporal server for the workflow tests (see the module docstring)."""
    test_server = os.environ.get("TEMPORAL_TEST_SERVER_PATH")
    if test_server:
        return await WorkflowEnvironment.start_time_skipping(test_server_existing_path=test_server)
    cli = os.environ.get("TEMPORAL_CLI_PATH")
    if cli:
        return await WorkflowEnvironment.start_local(dev_server_existing_path=cli)
    try:
        return await WorkflowEnvironment.start_time_skipping()
    except RuntimeError as e:
        if os.environ.get("CI"):
            raise  # never skip silently in CI: these tests must run there
        pytest.skip(f"No Temporal server (set TEMPORAL_TEST_SERVER_PATH or TEMPORAL_CLI_PATH): {e}")


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
    CALLS.clear()
    CALLS.update(llm=0, tool=0)
    ATTEMPTS.clear()
    SPANS.clear()
    yield
    for agent in (analyst, finance, finance_approval, cited, broken, thinker, rejected, throttled, patient):
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
        async with await _env() as env:
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
        async with await _env() as env:
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


def test_tool_return_metadata_crosses_the_activity():
    async def go():
        async with await _env() as env:
            opts = _options(CitedWorkflow)
            async with Worker(env.client, task_queue="q", **opts):
                out = await env.client.execute_workflow(
                    CitedWorkflow.run,
                    "Taux EUR ?",
                    id="wf-cited",
                    task_queue="q",
                    execution_timeout=timedelta(seconds=40),
                )
                history = await env.client.get_workflow_handle("wf-cited").fetch_history()
                await Replayer(workflows=opts["workflows"], workflow_runner=opts["workflow_runner"]).replay_workflow(
                    history
                )
                return out

    content, tool_content, metadata = asyncio.run(go())

    # The model reads the value only; the metadata reaches the application.
    assert content == "1 EUR = 1.08 USD"
    assert tool_content == "1.08"
    assert metadata == {"source": "ECB", "currency": "EUR"}
    assert CALLS["tool"] == 1


def test_agent_workflow_tool_approval_by_query_and_signal():
    async def go():
        async with await _env() as env:
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
        async with await _env() as env:
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
        async with await _env() as env:
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


# ---------------------------------------------------------------------------
# Bounded retries and unsupported options (no server)
# ---------------------------------------------------------------------------


def test_unsupported_options_are_refused_by_the_worker_not_at_construction():
    # `lead` and `auto_window` were built at import: a durability changes nothing outside a workflow.
    with pytest.raises(GwenflowDurabilityError, match="team"):
        worker_options(LeadWorkflow, agents=[lead])
    with pytest.raises(GwenflowDurabilityError, match="context_window='auto'"):
        worker_options(LeadWorkflow, agents=[auto_window])
    worker_options(FinanceWorkflow)  # a supported agent passes


def test_every_activity_has_bounded_retries():
    durability = finance.durability
    assert durability._config()["retry_policy"].maximum_attempts == 3
    assert durability._config_for_model()["retry_policy"].maximum_attempts == 10
    # A tool keeps its own attempt count; a tool that requires approval runs once.
    assert durability._config_for_tool("get_currency_rate")["retry_policy"].maximum_attempts == 3
    assert durability._config_for_tool("get_currency_rate", 2)["retry_policy"].maximum_attempts == 2
    assert finance_approval.durability._config_for_tool("get_currency_rate", 1)["retry_policy"].maximum_attempts == 1
    # An explicit policy still wins.
    custom = TemporalDurability(tool_activity_config={"search": {"retry_policy": RetryPolicy(maximum_attempts=7)}})
    assert custom._config_for_tool("search", 2)["retry_policy"].maximum_attempts == 7


def _model_request(agent):
    return next(a for a in agent.durability.activities if a.__name__ == "model_request")


def test_a_rejected_model_request_is_not_retried():
    request = {"messages": [{"role": "user", "content": "hi"}], "tool_names": [], "tool_choice": None}
    request["structured_output"] = False

    with pytest.raises(ApplicationError) as rejected_error:
        asyncio.run(_model_request(rejected)(request))
    assert rejected_error.value.non_retryable
    assert "400" in str(rejected_error.value)

    # 429 is transient: the provider error goes through, Temporal retries it.
    with pytest.raises(ProviderError):
        asyncio.run(_model_request(throttled)(request))

    # The provider adapters wrap the SDK error in a `RuntimeError`: its cause carries the status.
    with pytest.raises(ApplicationError) as wrapped_error:
        asyncio.run(_model_request(wrapped_rejected)(request))
    assert wrapped_error.value.non_retryable
    assert wrapped_error.value.type == "ProviderError"
    assert "401" in str(wrapped_error.value)
    with pytest.raises(RuntimeError) as transient_error:
        asyncio.run(_model_request(wrapped_throttled)(request))
    assert not isinstance(transient_error.value, ApplicationError)


def test_a_request_rejected_by_a_real_provider_client_is_not_retried():
    """Through `ChatOpenAI` and the openai SDK, against a local endpoint answering 401."""
    server = HTTPServer(("127.0.0.1", 0), _Unauthorized)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        agent = Agent(
            name="unauthorized",
            llm=ChatOpenAI(
                model="gpt-4o-mini",
                api_key="sk-test",
                base_url=f"http://127.0.0.1:{server.server_port}/v1",
                context_window=128000,
            ),
            history=_history(),
            durability=TemporalDurability(),
        )
        request = {"messages": [{"role": "user", "content": "hi"}], "tool_names": [], "tool_choice": None}
        request["structured_output"] = False
        with pytest.raises(ApplicationError) as error:
            asyncio.run(_model_request(agent)(request))
    finally:
        server.shutdown()
        server.server_close()

    assert error.value.non_retryable
    assert error.value.type == "AuthenticationError"
    assert "401" in str(error.value)
    assert CALLS["http"] == 1


# One test per provider SDK, with the SDK's own exception classes: the attribute that
# carries the HTTP status differs (`status_code`, or `code` for google-genai), and the
# adapters hand the activity a `RuntimeError` whose cause is the SDK error.


def _http_response(status: int) -> httpx.Response:
    return httpx.Response(status, request=httpx.Request("POST", "https://provider.test/v1/chat"), json={})


def _through_the_activity(error: Exception) -> BaseException:
    """What the `model_request` activity raises when the adapter fails with `error`."""
    agent = Agent(
        name="sdk-error",
        llm=AdapterChat(model="gpt-5-mini", context_window=128000, error=error),
        history=_history(),
        durability=TemporalDurability(),
    )
    request = {"messages": [{"role": "user", "content": "hi"}], "tool_names": [], "tool_choice": None}
    request["structured_output"] = False
    with pytest.raises(Exception) as raised:  # noqa: PT011 - the type is what each test checks
        asyncio.run(_model_request(agent)(request))
    return raised.value


def _check_sdk_errors(rejected: dict, transient: list) -> None:
    for status, error in rejected.items():
        assert _non_retryable_status(error) == status
        failure = _through_the_activity(error)
        assert isinstance(failure, ApplicationError), error
        assert failure.non_retryable
        assert failure.type == type(error).__name__
        assert str(status) in str(failure)
    for error in transient:
        assert _non_retryable_status(error) is None
        failure = _through_the_activity(error)
        assert type(failure) is RuntimeError, error  # left to Temporal's retry policy
        assert failure.__cause__ is error


def test_openai_sdk_errors():
    import openai

    def error(cls, status):
        return cls(f"HTTP {status}", response=_http_response(status), body={"code": "some_string_code"})

    _check_sdk_errors(
        rejected={
            400: error(openai.BadRequestError, 400),
            401: error(openai.AuthenticationError, 401),
            403: error(openai.PermissionDeniedError, 403),
            404: error(openai.NotFoundError, 404),
            422: error(openai.UnprocessableEntityError, 422),
        },
        transient=[
            error(openai.ConflictError, 409),
            error(openai.RateLimitError, 429),
            error(openai.InternalServerError, 500),
            openai.APIConnectionError(request=httpx.Request("POST", "https://provider.test")),
            openai.APITimeoutError(request=httpx.Request("POST", "https://provider.test")),
        ],
    )


def test_anthropic_sdk_errors():
    import anthropic

    def error(cls, status):
        return cls(f"HTTP {status}", response=_http_response(status), body=None)

    _check_sdk_errors(
        rejected={
            400: error(anthropic.BadRequestError, 400),
            401: error(anthropic.AuthenticationError, 401),
            403: error(anthropic.PermissionDeniedError, 403),
            404: error(anthropic.NotFoundError, 404),
            413: error(anthropic.APIStatusError, 413),
        },
        transient=[
            error(anthropic.RateLimitError, 429),
            error(anthropic.InternalServerError, 500),
            error(anthropic.APIStatusError, 529),  # overloaded
            anthropic.APIConnectionError(request=httpx.Request("POST", "https://provider.test")),
        ],
    )


def test_mistral_sdk_errors():
    from mistralai.client.errors import HTTPValidationError, MistralError, SDKError

    def error(status):
        return SDKError("API error occurred", _http_response(status))

    # Every HTTP error of the SDK gets its `status_code` from `MistralError`.
    assert issubclass(SDKError, MistralError) and issubclass(HTTPValidationError, MistralError)
    _check_sdk_errors(
        rejected={400: error(400), 401: error(401), 404: error(404), 422: error(422)},
        transient=[error(408), error(429), error(500), error(503)],
    )


def test_google_genai_sdk_errors():
    from google.genai import errors

    def body(status):
        return {"error": {"code": status, "message": f"HTTP {status}", "status": "SOME_STATUS"}}

    _check_sdk_errors(
        rejected={
            400: errors.ClientError(400, body(400)),
            403: errors.ClientError(403, body(403)),
            404: errors.ClientError(404, body(404)),
        },
        transient=[
            errors.ClientError(429, body(429)),
            errors.ServerError(500, body(500)),
            errors.ServerError(503, body(503)),
        ],
    )


# ---------------------------------------------------------------------------
# Workflows in flight (no server)
# ---------------------------------------------------------------------------

# Histories recorded with the code of commit fb5436b, before the bounded retries, the
# `reasoning_request` activity and the heartbeats (`tests/fixtures/temporal_histories`).
# A workflow started with that code must go on with this one: its history is replayed
# here, and a command that differs from the recorded one is a non-determinism error.
RECORDED_HISTORIES = {
    "finance": "FinanceWorkflow",
    "cited": "CitedWorkflow",
    "approval": "ApprovalWorkflow",
    "morning_brief": "MorningBrief",
    "durable_flow": "DurableFlow",
    "durable_flow_failed": "DurableFlow",
    "failing_flow": "FailingFlowWorkflow",
    "catching_flow": "CatchingFlowWorkflow",
}


@pytest.mark.parametrize("name", sorted(RECORDED_HISTORIES))
def test_a_history_recorded_before_the_fixes_still_replays(name):
    workflow_class = globals()[RECORDED_HISTORIES[name]]
    text = (FIXTURES / "temporal_histories" / f"{name}.json").read_text()
    history = WorkflowHistory.from_json(name, text)
    opts = _options(workflow_class)

    async def go():
        replayer = Replayer(
            workflows=opts["workflows"],
            workflow_runner=opts["workflow_runner"],
            workflow_failure_exception_types=opts["workflow_failure_exception_types"],
        )
        return await replayer.replay_workflow(history, raise_on_replay_failure=False)

    result = asyncio.run(go())
    assert result.replay_failure is None
    assert CALLS == {"llm": 0, "tool": 0}  # replayed from the history: no model, no tool


# ---------------------------------------------------------------------------
# Bounded retries in a workflow
# ---------------------------------------------------------------------------


def test_a_failing_tool_reaches_the_model_after_its_attempts():
    async def go():
        async with await _env() as env:
            async with Worker(env.client, task_queue="q", **_options(BrokenToolWorkflow)):
                return await env.client.execute_workflow(
                    BrokenToolWorkflow.run,
                    "Taux EUR ?",
                    id="wf-broken",
                    task_queue="q",
                    execution_timeout=timedelta(seconds=60),
                )

    content = asyncio.run(go())

    # As outside Temporal: the model reads the error instead of the run hanging.
    assert "no rate for EUR" in content
    assert CALLS["tool"] == 2


def test_a_rejected_model_request_fails_the_run():
    async def go():
        async with await _env() as env:
            async with Worker(env.client, task_queue="q", **_options(RejectedWorkflow)):
                return await env.client.execute_workflow(
                    RejectedWorkflow.run,
                    "Taux EUR ?",
                    id="wf-rejected",
                    task_queue="q",
                    execution_timeout=timedelta(seconds=60),
                )

    with pytest.raises(WorkflowFailureError):
        asyncio.run(go())
    assert CALLS["llm"] == 1


def test_a_long_model_request_reports_it_is_alive():
    async def go():
        async with await _env() as env:
            async with Worker(env.client, task_queue="q", **_options(PatientWorkflow)):
                return await env.client.execute_workflow(
                    PatientWorkflow.run,
                    "Take your time",
                    id="wf-patient",
                    task_queue="q",
                    execution_timeout=timedelta(seconds=60),
                )

    # Without heartbeats, the activity times out after 1 s and the run fails.
    assert asyncio.run(go()) == "took my time"
    assert CALLS["llm"] == 1


def test_a_reasoning_model_runs_as_an_activity():
    # Outside a workflow, nothing changes.
    assert asyncio.run(thinker.arun("Taux EUR ?")).content == "1 EUR = 1.08 USD"
    assert CALLS == {"llm": 2, "tool": 1, "reasoning": 1}
    thinker.history.reset()
    CALLS.clear()
    CALLS.update(llm=0, tool=0)

    async def go():
        async with await _env() as env:
            opts = _options(ThinkerWorkflow)
            async with Worker(env.client, task_queue="q", **opts):
                out = await env.client.execute_workflow(
                    ThinkerWorkflow.run,
                    "Taux EUR ?",
                    id="wf-thinker",
                    task_queue="q",
                    execution_timeout=timedelta(seconds=60),
                )
                history = await env.client.get_workflow_handle("wf-thinker").fetch_history()
                calls = dict(CALLS)
                await Replayer(workflows=opts["workflows"], workflow_runner=opts["workflow_runner"]).replay_workflow(
                    history
                )
                return out, _activity_types(history), calls

    content, activities, calls = asyncio.run(go())

    assert content == "1 EUR = 1.08 USD"
    assert activities == [
        "gwenflow__thinker__reasoning_request",
        "gwenflow__thinker__model_request",
        "gwenflow__thinker__call_tool",
        "gwenflow__thinker__model_request",
    ]
    assert calls == {"llm": 2, "tool": 1, "reasoning": 1}
    assert CALLS == calls  # the replay called no model and no tool


def test_an_unsupported_agent_fails_the_run_instead_of_hanging():
    async def go():
        async with await _env() as env:
            opts = _options(LeadWorkflow)
            opts["activities"] = [*opts["activities"], *lead.durability.activities]
            async with Worker(env.client, task_queue="q", **opts):
                return await env.client.execute_workflow(
                    LeadWorkflow.run,
                    "Taux EUR ?",
                    id="wf-lead",
                    task_queue="q",
                    execution_timeout=timedelta(seconds=30),
                )

    with pytest.raises(WorkflowFailureError) as failure:
        asyncio.run(go())
    assert "team" in str(failure.value.cause)
