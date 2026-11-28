"""Live checks of durable agents against a real Temporal dev server.

Not collected by the default test run (the file name does not start with
`test_`): they take about 45 s of real time, since a dev server does not skip
time, and `WorkflowEnvironment.start_local()` downloads the Temporal CLI on
first use. Run them on purpose:

    uv run pytest tests/temporal_live_checks.py --record-mode=none

What they check, each of which was a bug:

- a model request longer than the default 30 s heartbeat timeout completes;
- a 401 answered to a real `ChatOpenAI` client fails the activity at once;
- a workflow cancelled while a tool runs ends cancelled.
"""

import asyncio
import json
import threading
import time
from dataclasses import dataclass
from datetime import timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

pytest.importorskip("temporalio")

from temporalio import workflow  # noqa: E402
from temporalio.client import WorkflowFailureError  # noqa: E402
from temporalio.common import RetryPolicy  # noqa: E402
from temporalio.exceptions import ApplicationError  # noqa: E402
from temporalio.testing import WorkflowEnvironment  # noqa: E402
from temporalio.worker import Worker  # noqa: E402
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner  # noqa: E402

with workflow.unsafe.imports_passed_through():
    from gwenflow import Agent, ChatOpenAI, Tool
    from gwenflow.durability import AgentWorkflow, TemporalDurability, worker_options
    from gwenflow.durability.temporal import DEFAULT_MODEL_ACTIVITY_CONFIG
    from gwenflow.llms.base import ChatBase
    from gwenflow.memory import ChatMemoryBuffer
    from gwenflow.types import ModelResponse
    from gwenflow.types.message import TextContent, ToolCall

CALLS = {"llm": 0, "tool": 0, "http": 0}


def _count_bytes(text) -> int:
    return len(str(text).encode())


def _history():
    return ChatMemoryBuffer(max_fraction=0.85, tokenizer_fn=_count_bytes)


class _Stub(ChatBase):
    def invoke(self, *a, **k):
        raise NotImplementedError

    def stream(self, *a, **k):
        raise NotImplementedError

    async def astream(self, *a, **k):
        raise NotImplementedError
        yield


@dataclass(kw_only=True)
class SlowChat(_Stub):
    response_format: object = None
    seconds: float = 35.0

    async def ainvoke(self, input=None, **kw):
        CALLS["llm"] += 1
        await asyncio.sleep(self.seconds)
        return ModelResponse(parts=[TextContent(content="done")])


@dataclass(kw_only=True)
class ToolThenAnswer(_Stub):
    response_format: object = None

    async def ainvoke(self, input=None, **kw):
        CALLS["llm"] += 1
        tools = [m for m in input if m.get("role") == "tool"]
        if not tools:
            return ModelResponse(parts=[ToolCall(id="call_1", name="slow_tool", arguments="{}")])
        return ModelResponse(parts=[TextContent(content=f"answer after: {tools[-1]['content']}")])


def slow_tool() -> str:
    """A slow tool."""
    CALLS["tool"] += 1
    time.sleep(8)
    return "ok"


slow = Agent(
    name="slow",
    llm=SlowChat(model="gpt-5-mini", context_window=128000),
    history=_history(),
    durability=TemporalDurability(
        model_activity_config={**DEFAULT_MODEL_ACTIVITY_CONFIG, "retry_policy": RetryPolicy(maximum_attempts=1)}
    ),
)
cancellable = Agent(
    name="cancellable",
    llm=ToolThenAnswer(model="gpt-5-mini", context_window=128000),
    tools=[Tool(slow_tool)],
    history=_history(),
    durability=TemporalDurability(),
)


@workflow.defn
class SlowWorkflow(AgentWorkflow):
    __gwenflow_agents__ = [slow]

    @workflow.run
    async def run(self, prompt: str) -> str:
        return (await self.run_agent(slow, prompt)).content


@workflow.defn
class CancellableWorkflow(AgentWorkflow):
    __gwenflow_agents__ = [cancellable]

    @workflow.run
    async def run(self, prompt: str) -> str:
        return (await self.run_agent(cancellable, prompt)).content


def _options(*classes):
    opts = worker_options(*classes)
    restrictions = opts["workflow_runner"].restrictions.with_passthrough_modules(__name__)
    opts["workflow_runner"] = SandboxedWorkflowRunner(restrictions=restrictions)
    return opts


# -- 1. a real provider client answering 401 ---------------------------------


class _Unauthorized(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802 (name set by `BaseHTTPRequestHandler`)
        CALLS["http"] += 1
        body = json.dumps({"error": {"message": "bad key", "type": "invalid_request_error", "code": "invalid_api_key"}})
        self.send_response(401)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body.encode())

    def log_message(self, *a):
        pass


def test_a_401_from_a_real_openai_client_is_not_retried():
    server = HTTPServer(("127.0.0.1", 0), _Unauthorized)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    agent = Agent(
        name="real",
        llm=ChatOpenAI(
            model="gpt-4o-mini",
            api_key="sk-x",
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
            context_window=128000,
        ),
        history=_history(),
        durability=TemporalDurability(),
    )
    activity_fn = next(a for a in agent.durability.activities if a.__name__ == "model_request")
    request = {"messages": [{"role": "user", "content": "hi"}], "tool_names": [], "tool_choice": None}
    request["structured_output"] = False
    try:
        with pytest.raises(ApplicationError) as error:
            asyncio.run(activity_fn(request))
    finally:
        server.shutdown()
        server.server_close()
    assert error.value.non_retryable
    assert error.value.type == "AuthenticationError"
    assert CALLS["http"] == 1


# -- 2. a model request longer than the default heartbeat timeout -------------


def test_a_35_s_model_request_outlives_the_30_s_heartbeat_timeout():
    CALLS.update(llm=0)

    async def go():
        async with await WorkflowEnvironment.start_local() as env:
            async with Worker(env.client, task_queue="q", **_options(SlowWorkflow)):
                started = time.monotonic()
                try:
                    out = await env.client.execute_workflow(
                        SlowWorkflow.run, "hi", id="wf-slow", task_queue="q", execution_timeout=timedelta(seconds=120)
                    )
                    return "completed", out, time.monotonic() - started
                except WorkflowFailureError as e:
                    cause = e.cause
                    chain = []
                    while cause is not None:
                        chain.append(f"{type(cause).__name__}: {cause}")
                        cause = cause.__cause__
                    return "failed", chain, time.monotonic() - started

    outcome, detail, seconds = asyncio.run(go())
    assert outcome == "completed", detail  # it failed at 30.0 s with "activity Heartbeat timeout"
    assert detail == "done"
    assert seconds >= 35
    assert CALLS["llm"] == 1  # not retried


# -- 3. cancelling a workflow while a tool runs -------------------------------


def test_a_workflow_cancelled_during_a_tool_call_ends_cancelled():
    CALLS.update(llm=0, tool=0)

    async def go():
        async with await WorkflowEnvironment.start_local() as env:
            async with Worker(env.client, task_queue="q", **_options(CancellableWorkflow)):
                handle = await env.client.start_workflow(
                    CancellableWorkflow.run,
                    "hi",
                    id="wf-cancel",
                    task_queue="q",
                    execution_timeout=timedelta(seconds=60),
                )
                await asyncio.sleep(2)
                await handle.cancel()
                try:
                    out = await handle.result()
                    outcome = ("completed", out)
                except WorkflowFailureError as e:
                    outcome = ("failed", f"{type(e.cause).__name__}: {e.cause}")
                status = (await handle.describe()).status
                return outcome, status.name

    outcome, status = asyncio.run(go())
    assert outcome == ("failed", "CancelledError: Workflow cancelled")
    assert status == "CANCELED"
    assert CALLS["llm"] == 1  # the cancelled tool call was not shown to the model as an error
