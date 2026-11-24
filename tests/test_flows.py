import asyncio
import time
from dataclasses import dataclass

import pytest

from gwenflow.flows import AutoFlow, Flow, FlowError, FlowNodeError, FlowRunner
from gwenflow.flows.handlers import node_type

CALLS: dict[str, int] = {}


@node_type("test.Sleep")
def _sleep(parameters, input_data):
    time.sleep(parameters.get("seconds", 0.3))
    return parameters.get("value", "ok")


@node_type("test.Join")
def _join(parameters, input_data):
    return "+".join(str(input_data[k]) for k in sorted(input_data))


@node_type("test.Flaky")
def _flaky(parameters, input_data):
    key = parameters["key"]
    CALLS[key] = CALLS.get(key, 0) + 1
    if CALLS[key] <= parameters.get("failures", 1):
        raise RuntimeError("transient")
    return f"ok after {CALLS[key]}"


@node_type("test.Echo")
def _echo(parameters, input_data):
    return parameters


def diamond(seconds=0.3):
    return {
        "name": "diamond",
        "nodes": [
            {"name": "a", "type": "test.Sleep", "parameters": {"seconds": seconds, "value": "A"}},
            {"name": "b", "type": "test.Sleep", "parameters": {"seconds": seconds, "value": "B"}},
            {"name": "join", "type": "test.Join"},
        ],
        "connections": {"a": [[{"node": "join"}]], "b": [[{"node": "join"}]]},
    }


def test_independent_branches_run_concurrently():
    start = time.perf_counter()
    results = FlowRunner(Flow.from_dict(diamond(0.4))).run()
    assert results["join"] == "A+B"
    assert time.perf_counter() - start < 0.75  # sequential would take 0.8s


def test_sinks_order_and_roundtrip():
    flow = Flow.from_dict(diamond())
    assert flow.topological_order() == ["a", "b", "join"]
    assert flow.sinks() == ["join"]
    assert Flow.from_dict(flow.to_dict()).parents() == flow.parents()


def test_cycle_is_rejected():
    data = diamond()
    data["connections"]["join"] = [[{"node": "a"}]]
    with pytest.raises(FlowError, match="cycle"):
        Flow.from_dict(data)


def test_unknown_node_and_duplicates_are_rejected():
    data = diamond()
    data["nodes"].append({"name": "a", "type": "test.Sleep"})
    data["connections"]["ghost"] = [[{"node": "a"}]]
    with pytest.raises(FlowError) as err:
        Flow.from_dict(data)
    assert "Duplicate" in str(err.value) and "ghost" in str(err.value)


def test_failure_raises_flow_node_error_instead_of_exiting():
    data = {
        "name": "f",
        "nodes": [{"name": "boom", "type": "test.Flaky", "parameters": {"key": "boom", "failures": 9}}],
    }
    with pytest.raises(FlowNodeError) as err:
        FlowRunner(Flow.from_dict(data)).run()
    assert err.value.node == "boom"


def test_retries_from_node_settings(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    data = {
        "name": "f",
        "nodes": [
            {
                "name": "x",
                "type": "test.Flaky",
                "parameters": {"key": "retry", "failures": 2},
                "settings": {"retries": 2},
            }
        ],
    }
    assert FlowRunner(Flow.from_dict(data)).run()["x"] == "ok after 3"


async def _no_sleep(*_args, **_kwargs):
    return None


def test_credentials_come_from_the_environment(monkeypatch):
    data = {
        "name": "f",
        "nodes": [
            {"name": "e", "type": "test.Echo", "parameters": {"user": "u"}, "credentials": {"password": "PG_PASS"}}
        ],
    }
    flow = Flow.from_dict(data)
    assert "password" not in str(flow.to_dict()["nodes"][0]["parameters"])
    with pytest.raises(FlowNodeError, match="PG_PASS"):
        FlowRunner(flow).run()
    monkeypatch.setenv("PG_PASS", "s3cret")
    assert FlowRunner(flow).run()["e"] == {"user": "u", "password": "s3cret"}


def test_autoflow_plans_a_flow(monkeypatch):
    from gwenflow import ChatOpenAI
    from gwenflow.tools.shell import ShellTool

    tool = ShellTool()
    plan = (
        '[{"name": "Research", "task": "Find facts", "tools": ["%s"], "depends_on": []},' % tool.name
        + ' {"name": "Write", "task": "Write it up", "tools": null, "depends_on": ["Research"]}]'
    )

    class _Response:
        content = f"```json\n{plan}\n```"

    llm = ChatOpenAI(model="gpt-5-mini", api_key="x")
    monkeypatch.setattr(type(llm), "invoke", lambda self, **kw: _Response())
    flow = AutoFlow(llm=llm, tools=[tool]).plan("A short bio")
    assert flow.topological_order() == ["Research", "Write"]
    research = flow.node("Research").parameters
    assert research["model"] == "openai/gpt-5-mini" and research["tools"] == ["shell"]


def test_flow_with_durability_runs_normally_outside_a_workflow():
    class _Recorder:
        bound = None

        def bind(self, target):
            self.bound = target

    durability = _Recorder()
    flow = Flow.from_dict(diamond(0.01), durability=durability)
    assert durability.bound is flow
    assert "durability" not in flow.to_dict()

    results = asyncio.run(flow.arun(on_status=lambda name, status: statuses.append((name, status))))
    assert results["join"] == "A+B"
    assert statuses[-2:] == [("join", "running"), ("join", "done")]


statuses: list = []


def test_autoflow_runs_with_custom_llm_and_tools(monkeypatch):
    import tiktoken

    from gwenflow.llms.base import ChatBase
    from gwenflow.tools import Tool
    from gwenflow.types import ModelResponse
    from gwenflow.types.message import TextContent

    class _Bytes:  # no network for tiktoken vocabularies in tests
        def encode(self, text, **kw):
            return list(text.encode())

        def decode(self, ids, **kw):
            return bytes(ids).decode(errors="ignore")

    monkeypatch.setattr(tiktoken, "get_encoding", lambda name: _Bytes())
    monkeypatch.setattr(tiktoken, "encoding_for_model", lambda name: _Bytes())

    plan = (
        '```json\n[{"name": "Research", "task": "Find", "tools": ["my_lookup"], "depends_on": []},'
        ' {"name": "Write", "task": "Write", "tools": null, "depends_on": ["Research"]}]\n```'
    )

    @dataclass(kw_only=True)
    class CustomLLM(ChatBase):
        def invoke(self, input=None, **kw):
            if "depends_on" in str(input):
                return ModelResponse(parts=[TextContent(content=plan)])
            return ModelResponse(parts=[TextContent(content="done: " + input[-1]["content"])])

        async def ainvoke(self, input=None, **kw):
            return self.invoke(input)

        def stream(self, *a, **k):
            raise NotImplementedError

        async def astream(self, *a, **k):
            raise NotImplementedError
            yield

    def my_lookup(q: str) -> str:
        """Custom lookup."""
        return "x"

    autoflow = AutoFlow(llm=CustomLLM(model="custom", context_window=128000), tools=[Tool(my_lookup)])
    assert "model" not in autoflow.plan("bio").node("Research").parameters  # not a built-in provider
    result = autoflow.run("bio")
    assert result.startswith("done: Write") and "Output from Research" in result
