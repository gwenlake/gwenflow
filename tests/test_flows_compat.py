"""Backward compatibility of flows with v1.5.1.

A flow written for v1.5.1 behaves as it did then, except for the differences
kept on purpose (last section of this file). Everything here is offline: custom
node types, the built-in file / SQLite / Python nodes, and test doubles for the
nodes that reach a service (HTTP, Postgres, OpenSearch, readers, agents).
"""

import asyncio
import contextvars
import datetime
import decimal
import json
import logging
import random
import sys
import threading
import time
import types

import pytest

import gwenflow.flows.flows as flows_module
from gwenflow.flows import AutoFlow, Flow, FlowError, FlowNodeError, FlowRunner
from gwenflow.flows.flows import Node, jsonable_result, node_execution
from gwenflow.flows.handlers import NODE_TYPE_REGISTRY, agent_llm, compose_task, node_type
from gwenflow.logger import LOGGER_NAME

LOG: list = []
REQUEST_ID: contextvars.ContextVar[str] = contextvars.ContextVar("compat_request_id", default="unset")


class Opaque:
    """A value that is not JSON serializable (a DataFrame, a client object...)."""

    def __init__(self, n):
        self.n = n


@node_type("compat.Value")
def _value(parameters, input_data):
    LOG.append(("value", parameters.get("value"), threading.current_thread() is threading.main_thread()))
    return parameters.get("value")


@node_type("compat.Echo")
def _echo(parameters, input_data):
    return {"inputs": input_data, "order": list(input_data or {})}


@node_type("compat.Types")
def _types(parameters, input_data):
    return {
        "when": datetime.datetime(2026, 1, 2, 3, 4, 5),
        "amount": decimal.Decimal("12.50"),
        "pair": (1, 2),
        "ids": {3},
    }


@node_type("compat.TypeNames")
def _type_names(parameters, input_data):
    value = next(iter(input_data.values()))
    return {k: type(v).__name__ for k, v in value.items()}


@node_type("compat.Opaque")
def _opaque(parameters, input_data):
    return Opaque(7)


@node_type("compat.UseOpaque")
def _use_opaque(parameters, input_data):
    return next(iter(input_data.values())).n + 1


@node_type("compat.WriteFile")
def _write_file(parameters, input_data):
    time.sleep(0.2)
    with open(parameters["file"], "w") as f:
        f.write("written")
    return "ok"


@node_type("compat.ReadFile")
def _read_file(parameters, input_data):
    with open(parameters["file"]) as f:
        return f.read()


@node_type("compat.Fail")
def _fail(parameters, input_data):
    LOG.append(("fail-attempt",))
    raise RuntimeError("boom")


@node_type("compat.Flaky")
def _flaky(parameters, input_data):
    LOG.append(("flaky-attempt",))
    if sum(1 for entry in LOG if entry == ("flaky-attempt",)) <= parameters.get("failures", 1):
        raise RuntimeError("transient")
    return "recovered"


@node_type("compat.Slow")
def _slow(parameters, input_data):
    time.sleep(parameters["seconds"])
    return "late"


@node_type("compat.Track")
def _track(parameters, input_data):
    """Records how many `compat.Track` nodes run at the same time."""
    with _track_lock:
        _running.append(parameters["name"])
        LOG.append(("running-together", len(_running)))
        LOG.append(("started", parameters["name"]))
    time.sleep(0.05)
    with _track_lock:
        _running.remove(parameters["name"])
    return parameters["name"]


@node_type("compat.RequestId")
def _request_id(parameters, input_data):
    return REQUEST_ID.get()


_track_lock = threading.Lock()
_running: list = []


def chain(*names):
    return {a: [[{"node": b, "type": "main", "index": 0}]] for a, b in zip(names, names[1:], strict=False)}


def tiny(**extra):
    return {"name": "tiny", "nodes": [{"name": "a", "type": "compat.Value", "parameters": {"value": 1}}], **extra}


@pytest.fixture(autouse=True)
def reset_state(monkeypatch):
    LOG.clear()
    _running.clear()
    # No waiting between two attempts of a node.
    monkeypatch.setattr(flows_module, "RETRY_INITIAL_INTERVAL", 0.0)


@pytest.fixture
def warnings_logged():
    """Messages logged at WARNING by Gwenflow (its logger does not propagate to pytest's)."""
    messages: list[str] = []

    class _Collect(logging.Handler):
        def emit(self, record):
            if record.levelno >= logging.WARNING:
                messages.append(record.getMessage())

    handler = _Collect()
    logger = logging.getLogger(LOGGER_NAME)
    logger.addHandler(handler)
    yield messages
    logger.removeHandler(handler)


# ---------------------------------------------------------------------------
# 1. Parallel by default, one by one with `settings.parallel: false`
# ---------------------------------------------------------------------------


def unconnected(tmp_path, **settings):
    file = str(tmp_path / "implicit.txt")
    return {
        "name": "implicit",
        "nodes": [
            {"name": "write", "type": "compat.WriteFile", "parameters": {"file": file}},
            {"name": "read", "type": "compat.ReadFile", "parameters": {"file": file}},
        ],
        **({"settings": settings} if settings else {}),
    }


def test_parallel_false_runs_nodes_one_by_one_as_v1_5_1_did(tmp_path):
    # v1.5.1: two nodes without a connection ran in declaration order.
    results = Flow.from_dict(unconnected(tmp_path, parallel=False)).run()
    assert results == {"write": "ok", "read": "written"}


def test_parallel_stays_the_default(tmp_path):
    # Kept on purpose: without the setting, the two nodes start together and `read` finds no file.
    with pytest.raises(FlowNodeError, match="Node 'read' failed"):
        Flow.from_dict(unconnected(tmp_path)).run()


def fan(parallel):
    names = ["n1", "n2", "n3", "n4"]
    return {
        "name": "fan",
        "nodes": [{"name": n, "type": "compat.Track", "parameters": {"name": n}} for n in names],
        "settings": {"parallel": parallel},
    }


def test_parallel_false_never_runs_two_nodes_at_once():
    Flow.from_dict(fan(False)).run()
    assert max(count for kind, count in LOG if kind == "running-together") == 1
    assert [name for kind, name in LOG if kind == "started"] == ["n1", "n2", "n3", "n4"]

    LOG.clear()
    Flow.from_dict(fan(True)).run()
    assert max(count for kind, count in LOG if kind == "running-together") > 1


def test_parallel_false_stops_at_the_first_failure():
    flow = Flow.from_dict(
        {
            "name": "stops",
            "nodes": [
                {"name": "first", "type": "compat.Value", "parameters": {"value": 1}},
                {"name": "bad", "type": "compat.Fail"},
                {"name": "never", "type": "compat.Value", "parameters": {"value": 3}},
            ],
            "settings": {"parallel": False},
        }
    )
    statuses: list = []
    with pytest.raises(FlowNodeError) as failure:
        asyncio.run(flow.arun(on_status=lambda name, status: statuses.append((name, status))))
    assert failure.value.node == "bad"
    assert statuses == [
        ("first", "running"),
        ("first", "done"),
        ("bad", "running"),
        ("bad", "failed"),
        ("never", "skipped"),
    ]
    assert [entry[1] for entry in LOG if entry[0] == "value"] == [1]


def test_parallel_is_validated():
    Flow.from_dict(tiny(settings={"parallel": False})).validate()
    Flow.from_dict(tiny(settings={"parallel": True})).validate()
    with pytest.raises(FlowError, match="`parallel` must be true or false"):
        Flow.from_dict(tiny(settings={"parallel": "no"}))


def random_flow(rng):
    """A random DAG whose declaration order is not a topological order, with several output ports."""
    count = rng.randint(1, 9)
    names = [f"n{i}" for i in range(count)]
    declared = names[:]
    rng.shuffle(declared)
    rank = {name: i for i, name in enumerate(names)}
    connections = {}
    for source in declared:
        targets = [t for t in declared if rank[t] > rank[source] and rng.random() < 0.3]
        rng.shuffle(targets)
        if not targets:
            continue
        ports = [[{"node": t} for t in targets]]
        if len(targets) > 1 and rng.random() < 0.3:  # a target reached through two ports
            ports = [[{"node": targets[0]}], [{"node": t} for t in targets]]
        connections[source] = ports
    sources = list(connections)
    rng.shuffle(sources)
    return Flow.from_dict(
        {
            "name": "random",
            "nodes": [{"name": name, "type": "compat.Value"} for name in declared],
            "connections": {source: connections[source] for source in sources},
        }
    )


def test_sequential_order_is_the_order_networkx_gave_in_v1_5_1():
    nx = pytest.importorskip("networkx")
    rng = random.Random(151)
    differs_from_topological_order = 0
    for _ in range(500):
        flow = random_flow(rng)
        graph = FlowRunner(flow).graph  # built as v1.5.1 built it
        reference = [graph.nodes[node_id]["data"].name for node_id in nx.topological_sort(graph)]
        assert flow.sequential_order() == reference
        assert sorted(flow.topological_order()) == sorted(reference)
        differs_from_topological_order += flow.topological_order() != reference
    # `topological_order()` (declaration order inside a generation) is not that order:
    # this is why `parallel: false` has an order of its own.
    assert differs_from_topological_order > 0


def test_sequential_order_differs_from_topological_order_on_a_small_graph():
    flow = Flow.from_dict(
        {
            "name": "small",
            "nodes": [{"name": n, "type": "compat.Track", "parameters": {"name": n}} for n in ("c", "a", "b", "d")],
            "connections": {"a": [[{"node": "b"}, {"node": "c"}]]},
            "settings": {"parallel": False},
        }
    )
    assert flow.sequential_order() == ["a", "d", "b", "c"]  # networkx: targets in connection order
    assert flow.topological_order() == ["a", "d", "c", "b"]  # declaration order
    flow.run()
    assert [name for kind, name in LOG if kind == "started"] == ["a", "d", "b", "c"]


def test_a_cycle_has_no_sequential_order():
    flow = Flow(
        id="f",
        name="cyclic",
        nodes=[Node(id="1", type="compat.Value", name="a"), Node(id="2", type="compat.Value", name="b")],
        connections={},
    )
    from gwenflow.flows.flows import ConnectionEndpoint

    flow.connections = {"1": [[ConnectionEndpoint(node="2")]], "2": [[ConnectionEndpoint(node="1")]]}
    with pytest.raises(FlowError, match="cycle"):
        flow.sequential_order()


# ---------------------------------------------------------------------------
# 1 (continued). `concurrent_pairs()` and `gwenflow check`
# ---------------------------------------------------------------------------


def pipeline(tmp_path):
    database = str(tmp_path / "db.sqlite")
    return {
        "name": "pairs",
        "nodes": [
            {"name": "load", "type": "compat.Value"},
            {
                "name": "orders",
                "type": "gwenflow.sqliteQuery",
                "parameters": {"database": database, "query": "insert into orders values (1)"},
            },
            {
                "name": "clients",
                "type": "gwenflow.sqliteQuery",
                "parameters": {"database": database, "query": "insert into clients values (1)"},
            },
            {"name": "report", "type": "gwenflow.jsonWriter", "parameters": {"file": str(tmp_path / "report.json")}},
            {"name": "audit", "type": "gwenflow.jsonWriter", "parameters": {"file": str(tmp_path / "audit.json")}},
        ],
        "connections": {
            "load": [[{"node": "orders"}, {"node": "clients"}]],
            "orders": [[{"node": "report"}]],
            "clients": [[{"node": "report"}]],
        },
    }


def test_concurrent_pairs_are_the_nodes_without_a_path_between_them(tmp_path):
    flow = Flow.from_dict(pipeline(tmp_path))
    # `audit` is connected to nothing; `orders` and `clients` are two branches of `load`.
    assert flow.concurrent_pairs() == [
        ("load", "audit"),
        ("orders", "clients"),
        ("orders", "audit"),
        ("clients", "audit"),
        ("report", "audit"),
    ]
    # A chain has none: every node waits for the previous one.
    linear = {
        "name": "linear",
        "nodes": [{"name": n, "type": "compat.Value"} for n in "abc"],
        "connections": chain("a", "b", "c"),
    }
    assert Flow.from_dict(linear).concurrent_pairs() == []


def test_concurrent_report_flags_the_pairs_that_declare_the_same_path(tmp_path):
    report = {
        (pair.first, pair.second): pair.shared_paths for pair in Flow.from_dict(pipeline(tmp_path)).concurrent_report()
    }
    assert report[("orders", "clients")] == (str(tmp_path / "db.sqlite"),)
    assert report[("orders", "audit")] == ()
    assert report[("report", "audit")] == ()  # two different files


def test_check_command_lists_the_pairs_of_a_flow_file(tmp_path):
    from click.testing import CliRunner

    from gwenflow.cli import main

    path = tmp_path / "pairs.json"
    path.write_text(json.dumps(pipeline(tmp_path)))
    result = CliRunner().invoke(main, ["check", str(path)], terminal_width=400)
    assert result.exit_code == 0, result.output
    assert f"  orders | clients  <- both declare {tmp_path / 'db.sqlite'}" in result.output.splitlines()
    assert "  report | audit" in result.output.splitlines()
    assert "1 pair(s) declare the same path" in result.output

    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps({"name": "bad", "nodes": [{"name": "a", "type": "compat.DoesNotExist"}]}))
    result = CliRunner().invoke(main, ["check", str(invalid)], terminal_width=400)
    assert result.exit_code == 1
    assert "unknown type" in result.output


# ---------------------------------------------------------------------------
# 2. `run()` from async code
# ---------------------------------------------------------------------------


def from_a_running_loop(fn):
    async def caller():
        return fn()

    return asyncio.run(caller())


def test_flow_run_works_from_a_running_event_loop():
    assert from_a_running_loop(lambda: Flow.from_dict(tiny()).run()) == {"a": 1}


def test_flow_runner_run_works_from_a_running_event_loop(capsys):
    assert from_a_running_loop(lambda: FlowRunner(Flow.from_dict(tiny())).run()) == {"a": 1}
    assert capsys.readouterr().out.splitlines().count("1") == 1


def test_run_from_a_running_event_loop_keeps_the_context_variables():
    flow = Flow.from_dict({"name": "context", "nodes": [{"name": "id", "type": "compat.RequestId"}]})

    async def endpoint():
        REQUEST_ID.set("request-42")
        return flow.run()

    assert asyncio.run(endpoint()) == {"id": "request-42"}


def test_run_from_a_running_event_loop_raises_the_flow_errors():
    flow = Flow.from_dict({"name": "failing", "nodes": [{"name": "bad", "type": "compat.Fail"}]})
    with pytest.raises(FlowNodeError, match="boom"):
        from_a_running_loop(flow.run)


# ---------------------------------------------------------------------------
# 3. Results are handed to the next node as they are
# ---------------------------------------------------------------------------


def test_python_types_reach_the_next_node_unchanged():
    flow = Flow.from_dict(
        {
            "name": "types",
            "nodes": [{"name": "src", "type": "compat.Types"}, {"name": "names", "type": "compat.TypeNames"}],
            "connections": chain("src", "names"),
        }
    )
    results = flow.run()
    assert results["names"] == {"when": "datetime", "amount": "Decimal", "pair": "tuple", "ids": "set"}
    assert results["src"]["amount"] == decimal.Decimal("12.50")


def test_a_non_json_object_reaches_the_next_node():
    flow = Flow.from_dict(
        {
            "name": "opaque",
            "nodes": [{"name": "src", "type": "compat.Opaque"}, {"name": "use", "type": "compat.UseOpaque"}],
            "connections": chain("src", "use"),
        }
    )
    results = flow.run()
    assert results["use"] == 8
    assert isinstance(results["src"], Opaque)


def test_the_temporal_activity_still_converts_results_to_json():
    pytest.importorskip("temporalio")
    from temporalio.exceptions import ApplicationError
    from temporalio.testing import ActivityEnvironment

    from gwenflow.durability.flows import FlowNodeActivity

    def run(type_name):
        node = Node(id="1", type=type_name, name="src")
        request = {"node": flows_module.asdict(node), "input_data": None}
        return asyncio.run(ActivityEnvironment().run(FlowNodeActivity().run, request))

    # Recorded in the workflow history: JSON data, as before.
    assert run("compat.Types") == {"when": "2026-01-02T03:04:05", "amount": "12.50", "pair": [1, 2], "ids": [3]}
    with pytest.raises(ApplicationError) as failure:
        run("compat.Opaque")
    assert failure.value.non_retryable
    assert failure.value.type == "TypeError"
    assert failure.value.message == (
        "node type 'compat.Opaque' returned a value that is not JSON serializable "
        "(a Opaque is not JSON serializable): return dicts, lists, strings, numbers, booleans or None"
    )
    with pytest.raises(TypeError, match="is not JSON serializable"):
        jsonable_result("compat.Opaque", Opaque(1))


# ---------------------------------------------------------------------------
# 4. Model of the agent node
# ---------------------------------------------------------------------------


@pytest.fixture
def provider_keys(monkeypatch):
    for name in ("OPENAI", "MISTRAL", "ANTHROPIC", "GWENLAKE", "GOOGLE"):
        monkeypatch.setenv(f"{name}_API_KEY", "x")


@pytest.fixture
def agent_runs(monkeypatch):
    """Agents answer without a model: each run records its client, model, task and tools."""
    import gwenflow.agents.agent as agent_module

    runs: list[dict] = []

    def fake_run(self, task, *args, **kwargs):
        runs.append(
            {
                "llm": type(self.llm).__name__,
                "model": self.llm.model,
                "task": task,
                "name": self.name,
                "thread": threading.current_thread() is threading.main_thread(),
            }
        )

        class Response:
            content = f"answer of {self.name}"

        return Response()

    monkeypatch.setattr(agent_module.Agent, "run", fake_run)
    return runs


@pytest.mark.parametrize(
    ("model", "client", "sent"),
    [
        (None, "ChatOpenAI", "gpt-4o-mini"),
        ("openai/gpt-4o-mini", "ChatOpenAI", "gpt-4o-mini"),
        ("gpt-4o-mini", "ChatOpenAI", "gpt-4o-mini"),
        # An open-weights name keeps its organization: v1.5.1 cut it, and the endpoint refused the model.
        ("Qwen/Qwen3-32B", "ChatOpenAI", "Qwen/Qwen3-32B"),
        ("meta-llama/Llama-3.3-70B-Instruct", "ChatOpenAI", "meta-llama/Llama-3.3-70B-Instruct"),
        ("mistralai/Mistral-Small-24B", "ChatOpenAI", "mistralai/Mistral-Small-24B"),
        ("openai/Qwen/Qwen3-32B", "ChatOpenAI", "Qwen/Qwen3-32B"),
        # A prefix that names a provider is removed, as in v1.5.1, and does not choose the client.
        ("mistral/mistral-large-latest", "ChatOpenAI", "mistral-large-latest"),
        ("gwenlake/qwen3", "ChatOpenAI", "qwen3"),
    ],
)
def test_without_provider_the_agent_node_uses_an_openai_client(model, client, sent, provider_keys, agent_runs):
    parameters = {"instructions": "be brief", "task": "Summarize {news}"}
    if model:
        parameters["model"] = model
    assert NODE_TYPE_REGISTRY["gwenflow.Agent"](parameters, {"news": "N"}) == "answer of Flow Agent"
    assert (agent_runs[0]["llm"], agent_runs[0]["model"]) == (client, sent)


@pytest.mark.parametrize(
    ("provider", "model", "client", "sent"),
    [
        ("mistral", "mistral-large-latest", "ChatMistral", "mistral-large-latest"),
        ("gwenlake", "qwen3", "ChatGwenlake", "qwen3"),
        ("anthropic", "claude-sonnet-4-5", "ChatAnthropic", "claude-sonnet-4-5"),
        ("openai", "gpt-4o-mini", "ChatOpenAI", "gpt-4o-mini"),
        # With `provider`, the model name is sent as written: nothing is cut at the `/`.
        ("openai", "Qwen/Qwen3-32B", "ChatOpenAI", "Qwen/Qwen3-32B"),
        ("OpenAI", "openai/gpt-oss-120b", "ChatOpenAI", "openai/gpt-oss-120b"),
    ],
)
def test_provider_chooses_the_client_of_the_agent_node(provider, model, client, sent, provider_keys):
    llm = agent_llm({"provider": provider, "model": model})
    assert (type(llm).__name__, llm.model) == (client, sent)


def test_provider_needs_a_model_and_a_known_name(provider_keys):
    with pytest.raises(ValueError, match="needs a `model`"):
        agent_llm({"provider": "mistral"})
    with pytest.raises(ValueError, match="Unknown provider 'qwen'"):
        agent_llm({"provider": "qwen", "model": "Qwen3-32B"})


def test_a_provider_prefix_without_provider_is_reported(provider_keys, warnings_logged):
    agent_llm({"model": "mistral/mistral-large-latest"})
    assert len(warnings_logged) == 1
    assert "mistral/mistral-large-latest" in warnings_logged[0] and "provider: mistral" in warnings_logged[0]

    warnings_logged.clear()
    agent_llm({"model": "google/gemma-3-27b-it"})  # an open-weights name whose organization is a provider's name
    assert "'gemma-3-27b-it'" in warnings_logged[0] and "provider: openai" in warnings_logged[0]

    warnings_logged.clear()
    for model in ("openai/gpt-4o-mini", "gpt-4o-mini", "Qwen/Qwen3-32B"):
        agent_llm({"model": model})
    assert warnings_logged == []


def test_provider_of_a_harness_agent_node(provider_keys, agent_runs, monkeypatch):
    import gwenflow.harness as harness_module

    class FakeHarness:
        class config:  # noqa: N801
            model = "harness-model"

        calls: list = []

        def build_agent(self, **kwargs):
            from gwenflow import Agent

            FakeHarness.calls.append(kwargs)
            return Agent(name="Harness Agent", llm=kwargs.get("llm") or agent_llm({"model": kwargs["model"] or "m"}))

    monkeypatch.setattr(harness_module.Harness, "load", classmethod(lambda cls, path: FakeHarness()))
    handler = NODE_TYPE_REGISTRY["gwenflow.Agent"]

    # Without `provider`: unchanged, the harness chooses its client and `model` overrides its model.
    handler({"harness": "support", "model": "qwen3"}, None)
    assert FakeHarness.calls[-1] == {"model": "qwen3"}

    # With `provider`: that provider's client, running the node's model, else the harness one.
    handler({"harness": "support", "provider": "mistral", "model": "mistral-small"}, None)
    llm = FakeHarness.calls[-1]["llm"]
    assert (type(llm).__name__, llm.model) == ("ChatMistral", "mistral-small")
    handler({"harness": "support", "provider": "mistral"}, None)
    assert FakeHarness.calls[-1]["llm"].model == "harness-model"


# ---------------------------------------------------------------------------
# 5. Parent outputs the task does not reference
# ---------------------------------------------------------------------------


def test_unreferenced_parent_outputs_are_left_out_by_default(provider_keys, agent_runs):
    inputs = {"news": "N", "extra": {"big": "payload"}}
    handler = NODE_TYPE_REGISTRY["gwenflow.Agent"]

    handler({"instructions": "be brief", "task": "Summarize {news}"}, inputs)
    assert agent_runs[-1]["task"] == "Summarize N"  # as in v1.5.1

    handler({"instructions": "be brief", "task": "Summarize {news}", "append_inputs": True}, inputs)
    assert agent_runs[-1]["task"] == (
        'Summarize N\n\nContext from previous steps:\nOutput from extra:\n{"big": "payload"}'
    )


def test_compose_task():
    assert compose_task("Use {a}", {"a": {"k": 1}, "b": "B"}) == 'Use {"k": 1}'
    assert compose_task("Use {a}", {"a": "A", "b": "B"}, append_inputs=True).endswith("Output from b:\nB")
    assert compose_task("No input", None, append_inputs=True) == "No input"


# ---------------------------------------------------------------------------
# 6. `FlowRunner.run()` prints the result of the last node
# ---------------------------------------------------------------------------


def linear():
    return {
        "name": "linear",
        "nodes": [
            {"name": "first", "type": "compat.Value", "parameters": {"value": "one"}},
            {"name": "last", "type": "compat.Value", "parameters": {"value": {"file": "out.json"}}},
        ],
        "connections": chain("first", "last"),
    }


def printed(capsys):
    """Lines written with `print` (the logger writes its own, starting with the level)."""
    return [line for line in capsys.readouterr().out.splitlines() if not line.startswith(("INFO", "WARNING", " "))]


def test_flow_runner_run_prints_the_last_result_and_returns_every_result(capsys):
    results = FlowRunner(Flow.from_dict(linear())).run()
    assert results == {"first": "one", "last": {"file": "out.json"}}  # kept on purpose: v1.5.1 returned None
    assert printed(capsys) == ["{'file': 'out.json'}"]


def test_only_flow_runner_run_prints(capsys):
    flow = Flow.from_dict(linear())
    flow.run()
    asyncio.run(flow.arun())
    asyncio.run(FlowRunner(flow).arun())
    assert printed(capsys) == []


def test_the_printed_result_is_the_one_of_the_last_node_v1_5_1_ran(capsys):
    flow = Flow.from_dict(
        {
            "name": "order",
            "nodes": [{"name": n, "type": "compat.Value", "parameters": {"value": n}} for n in ("c", "a", "b", "d")],
            "connections": {"a": [[{"node": "b"}, {"node": "c"}]]},
        }
    )
    FlowRunner(flow).run()
    assert printed(capsys) == ["c"]  # networkx order: a, d, b, c


def test_the_file_given_to_flow_runner_can_be_a_str_or_a_path(tmp_path, capsys):
    path = tmp_path / "linear.json"
    path.write_text(json.dumps(linear()))
    assert FlowRunner(str(path)).run() == FlowRunner(path).run()
    assert len(FlowRunner(path).graph.nodes) == 2


# ---------------------------------------------------------------------------
# 7. A `credentials` block that names no environment variable is ignored
# ---------------------------------------------------------------------------


def test_a_credentials_block_from_another_tool_is_ignored_with_a_warning(warnings_logged):
    flow = Flow.from_dict(
        {
            "name": "extra",
            "description": "ignored key",
            "nodes": [
                {
                    "name": "a",
                    "type": "compat.Value",
                    "parameters": {"value": 1},
                    "credentials": {"api": {"id": "1", "name": "my key"}},  # n8n-style
                    "notes": "free text",
                }
            ],
        }
    )
    assert flow.node("a").credentials == {}
    assert flow.run() == {"a": 1}
    assert len(warnings_logged) == 1
    assert "node 'a'" in warnings_logged[0] and "api" in warnings_logged[0]


def test_credentials_that_name_an_environment_variable_still_work(monkeypatch, warnings_logged):
    monkeypatch.setenv("COMPAT_TOKEN", "s3cret")
    flow = Flow.from_dict(
        {
            "name": "mixed",
            "nodes": [
                {
                    "name": "a",
                    "type": "compat.Echo",
                    "credentials": {"token": "COMPAT_TOKEN", "empty": "", "other": {"id": "1"}},
                }
            ],
        }
    )
    assert flow.node("a").credentials == {"token": "COMPAT_TOKEN"}
    assert "empty, other" in warnings_logged[0]
    flow.run()

    # A block that is not a mapping is still a mistake in the file.
    with pytest.raises(FlowError, match="`credentials` must be a mapping"):
        Flow.from_dict({"name": "bad", "nodes": [{"name": "a", "type": "compat.Value", "credentials": ["X"]}]})


# ---------------------------------------------------------------------------
# 8. `FlowRunner._execute_node`
# ---------------------------------------------------------------------------


class Audited(FlowRunner):
    def _execute_node(self, node, input_data):
        LOG.append(("audited", node.name, input_data))
        return super()._execute_node(node, input_data)


def test_an_execute_node_override_is_called_for_every_node(capsys):
    assert hasattr(FlowRunner(Flow.from_dict(tiny())), "_execute_node")
    flow = Flow.from_dict(
        {
            "name": "audited",
            "nodes": [
                {"name": "a", "type": "compat.Value", "parameters": {"value": "A"}},
                {"name": "join", "type": "compat.Echo"},
            ],
            "connections": chain("a", "join"),
        }
    )
    results = Audited(flow).run()
    assert results["join"] == {"inputs": {"a": "A"}, "order": ["a"]}
    assert [entry[1:] for entry in LOG if entry[0] == "audited"] == [("a", None), ("join", {"a": "A"})]

    LOG.clear()
    asyncio.run(Audited(flow).arun())
    assert [entry[1] for entry in LOG if entry[0] == "audited"] == ["a", "join"]


def test_an_execute_node_override_can_replace_the_node():
    class Stubbed(FlowRunner):
        def _execute_node(self, node, input_data):
            return f"stub of {node.name}"

    flow = Flow.from_dict({"name": "stubbed", "nodes": [{"name": "bad", "type": "compat.Fail"}]})
    assert asyncio.run(Stubbed(flow).arun()) == {"bad": "stub of bad"}
    assert LOG == []


def test_retries_apply_around_an_execute_node_override():
    flow = Flow.from_dict(
        {
            "name": "retried",
            "nodes": [
                {"name": "flaky", "type": "compat.Flaky", "parameters": {"failures": 2}, "settings": {"retries": 2}}
            ],
        }
    )
    assert asyncio.run(Audited(flow).arun()) == {"flaky": "recovered"}
    assert sum(1 for entry in LOG if entry[0] == "audited") == 3  # once per attempt


def test_the_timeout_applies_around_an_execute_node_override():
    flow = Flow.from_dict(
        {
            "name": "slow",
            "nodes": [
                {"name": "slow", "type": "compat.Slow", "parameters": {"seconds": 1.5}, "settings": {"timeout": 0.2}}
            ],
        }
    )
    with pytest.raises(FlowNodeError, match="timed out after 0.2s"):
        asyncio.run(Audited(flow).arun())
    assert [entry[1] for entry in LOG if entry[0] == "audited"] == ["slow"]  # not retried: it may still be running


def test_execute_node_resolves_the_credentials(monkeypatch):
    monkeypatch.setenv("COMPAT_TOKEN", "s3cret")
    flow = Flow.from_dict(
        {
            "name": "creds",
            "nodes": [{"name": "a", "type": "test_compat.Parameters", "credentials": {"token": "COMPAT_TOKEN"}}],
        }
    )
    assert asyncio.run(Audited(flow).arun()) == {"a": {"token": "s3cret"}}


@node_type("test_compat.Parameters")
def _parameters(parameters, input_data):
    return parameters


# ---------------------------------------------------------------------------
# Built-in nodes the scenarios script does not cover: what changed since v1.5.1
# ---------------------------------------------------------------------------

# Their code is the one of v1.5.1 but for three things: the node types that only read
# (or that write a whole file) are retried, a node `timeout` bounds the call, and a
# failed Python script raises a `RuntimeError` that carries its stderr.


@pytest.mark.parametrize(
    ("type_name", "parameters", "retries"),
    [
        ("gwenflow.httpRequest", {"url": "https://x.test"}, 2),  # GET
        ("gwenflow.httpRequest", {"url": "https://x.test", "method": "head"}, 2),
        ("gwenflow.httpRequest", {"url": "https://x.test", "method": "POST"}, 0),
        ("gwenflow.httpRequest", {"url": "https://x.test", "method": "DELETE"}, 0),
        ("gwenflow.postgresQuery", {"query": "select 1"}, 0),
        ("gwenflow.sqliteQuery", {"query": "select 1"}, 0),
        ("gwenflow.opensearchQuery", {"operation": "search"}, 2),
        ("gwenflow.opensearchQuery", {}, 2),
        ("gwenflow.opensearchQuery", {"operation": "count"}, 2),
        ("gwenflow.opensearchQuery", {"operation": "get"}, 2),
        ("gwenflow.opensearchQuery", {"operation": "put"}, 0),
        ("gwenflow.opensearchQuery", {"operation": "delete"}, 0),
        ("gwenflow.pdfReader", {}, 2),
        ("gwenflow.excelReader", {}, 2),
        ("gwenflow.csvReader", {}, 2),
        ("gwenflow.jsonReader", {}, 2),
        ("gwenflow.websiteReader", {}, 2),
        ("gwenflow.csvWriter", {}, 2),
        ("gwenflow.jsonWriter", {}, 2),
        ("gwenflow.textSplitter", {}, 2),
        ("gwenflow.Python", {}, 0),
        ("gwenflow.Agent", {}, 0),
        ("compat.Value", {}, 0),  # a custom node type is not retried unless it says it is idempotent
    ],
)
def test_which_built_in_nodes_are_retried(type_name, parameters, retries):
    node = Node(id="1", type=type_name, name="n", parameters=parameters)
    assert node_execution(node).retries == retries
    # The flow or the node can always switch retries off.
    assert (
        node_execution(Node(id="1", type=type_name, name="n", parameters=parameters, settings={"retries": 0})).retries
        == 0
    )


class FakeHttpResponse:
    def __init__(self, payload):
        self.payload = payload
        self.text = "plain text"

    def raise_for_status(self):
        pass

    def json(self):
        if self.payload is None:
            raise ValueError("not json")
        return self.payload


def http_flow(**parameters):
    node = {"name": "call", "type": "gwenflow.httpRequest", "parameters": {"url": "https://x.test/api", **parameters}}
    return Flow.from_dict({"name": "http", "nodes": [node]})


def test_http_node_sends_the_same_request_as_v1_5_1(monkeypatch):
    import httpx

    calls: list = []

    def fake_request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        return FakeHttpResponse({"ok": True})

    monkeypatch.setattr(httpx, "request", fake_request)
    flow = http_flow(
        method="post",
        headers={"X-A": "1"},
        query_params={"q": "x"},
        body={"k": 1},
        auth={"type": "bearer", "token": "t0k"},
    )
    assert flow.run() == {"call": {"ok": True}}
    assert calls == [
        (
            "POST",
            "https://x.test/api",
            {
                "headers": {"X-A": "1", "Authorization": "Bearer t0k"},
                "params": {"q": "x"},
                "json": {"k": 1},
                "timeout": 30,  # the default of v1.5.1, when the node has no `timeout` setting
            },
        )
    ]

    calls.clear()
    monkeypatch.setattr(httpx, "request", lambda *a, **k: FakeHttpResponse(None))
    assert http_flow(auth={"type": "basic", "username": "u", "password": "p"}).run() == {"call": "plain text"}


def test_http_node_retries_a_get_but_not_a_post(monkeypatch):
    import httpx

    attempts: list = []

    def failing_request(method, url, **kwargs):
        attempts.append(method)
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "request", failing_request)
    with pytest.raises(FlowNodeError, match="connection refused"):
        http_flow().run()
    assert attempts == ["GET", "GET", "GET"]  # kept on purpose: 1 attempt in v1.5.1

    attempts.clear()
    with pytest.raises(FlowNodeError):
        http_flow(method="POST").run()
    assert attempts == ["POST"]

    attempts.clear()
    flow = Flow.from_dict(
        {
            "name": "http",
            "nodes": [
                {
                    "name": "call",
                    "type": "gwenflow.httpRequest",
                    "parameters": {"url": "https://x.test"},
                    "settings": {"retries": 0},
                }
            ],
        }
    )
    with pytest.raises(FlowNodeError):
        flow.run()
    assert attempts == ["GET"]  # `retries: 0` gives the behaviour of v1.5.1


def test_http_node_timeout_is_bounded_by_the_node_timeout(monkeypatch):
    import httpx

    seen: list = []
    monkeypatch.setattr(httpx, "request", lambda *a, **k: seen.append(k["timeout"]) or FakeHttpResponse({}))
    flow = Flow.from_dict(
        {
            "name": "http",
            "nodes": [
                {
                    "name": "call",
                    "type": "gwenflow.httpRequest",
                    "parameters": {"url": "https://x.test"},
                    "settings": {"timeout": 5},
                }
            ],
        }
    )
    flow.run()
    assert 0 < seen[0] <= 5


class FakeCursor:
    def __init__(self, connection):
        self.connection = connection
        self.description = None
        self.rowcount = 1

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, query, params):
        self.connection.statements.append((query, list(params)))
        self.description = ("a",) if query.lstrip().lower().startswith("select") else None

    def fetchall(self):
        return [{"a": 1}]


class FakeConnection:
    def __init__(self, conn_string, row_factory):
        self.conn_string = conn_string
        self.statements: list = []
        self.committed = False

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, statement):
        self.statements.append(statement)

    def cursor(self):
        return FakeCursor(self)

    def commit(self):
        self.committed = True


@pytest.fixture
def fake_psycopg(monkeypatch):
    connections: list = []

    def connect(conn_string, row_factory=None):
        connections.append(FakeConnection(conn_string, row_factory))
        return connections[-1]

    psycopg = types.ModuleType("psycopg")
    psycopg.connect = connect
    rows = types.ModuleType("psycopg.rows")
    rows.dict_row = object()
    psycopg.rows = rows
    monkeypatch.setitem(sys.modules, "psycopg", psycopg)
    monkeypatch.setitem(sys.modules, "psycopg.rows", rows)
    return connections


def postgres_flow(query, settings=None):
    node = {
        "name": "pg",
        "type": "gwenflow.postgresQuery",
        "parameters": {"host": "db", "database": "shop", "user": "u", "password": "p", "query": query, "params": [1]},
    }
    if settings:
        node["settings"] = settings
    return Flow.from_dict({"name": "postgres", "nodes": [node]})


def test_postgres_node_runs_the_same_statements_as_v1_5_1(fake_psycopg):
    assert postgres_flow("select * from t where a = %s").run() == {"pg": [{"a": 1}]}
    connection = fake_psycopg[-1]
    assert connection.conn_string == "host=db port=5432 dbname=shop user=u password=p"
    assert connection.statements == [("select * from t where a = %s", [1])]  # nothing added to the session

    assert postgres_flow("delete from t where a = %s").run() == {"pg": {"affected_rows": 1}}
    assert fake_psycopg[-1].committed


def test_postgres_node_sets_a_statement_timeout_only_with_a_node_timeout(fake_psycopg):
    postgres_flow("select 1", settings={"timeout": 5}).run()
    first, second = fake_psycopg[-1].statements
    assert first.startswith("SET statement_timeout = ")
    assert 0 < int(first.rsplit(" ", 1)[1]) <= 5000
    assert second == ("select 1", [1])


def test_a_write_to_postgres_is_not_retried(fake_psycopg, monkeypatch):
    attempts: list = []

    def failing_connect(conn_string, row_factory=None):
        attempts.append(conn_string)
        raise ConnectionError("server closed the connection")

    monkeypatch.setattr(sys.modules["psycopg"], "connect", failing_connect)
    with pytest.raises(FlowNodeError, match="server closed the connection"):
        postgres_flow("insert into t values (%s)").run()
    assert len(attempts) == 1


class FakeOpenSearchStore:
    instances: list = []
    failures = 0

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls: list = []
        FakeOpenSearchStore.instances.append(self)

    def search(self, **kwargs):
        self.calls.append(("search", kwargs))
        if FakeOpenSearchStore.failures:
            FakeOpenSearchStore.failures -= 1
            raise ConnectionError("node unavailable")
        return [{"id": "1"}]

    def count(self, query):
        return 3

    def get(self, id):
        return {"id": id}

    def put(self, data, column_id="id"):
        self.calls.append(("put", data, column_id))
        if FakeOpenSearchStore.failures:
            FakeOpenSearchStore.failures -= 1
            raise ConnectionError("node unavailable")
        return True

    def delete(self, ids):
        self.calls.append(("delete", ids))
        return True


@pytest.fixture
def fake_opensearch(monkeypatch):
    module = types.ModuleType("gwenflow.stores.opensearch")
    module.OpenSearchDocumentStore = FakeOpenSearchStore
    monkeypatch.setitem(sys.modules, "gwenflow.stores.opensearch", module)
    FakeOpenSearchStore.instances.clear()
    FakeOpenSearchStore.failures = 0
    return FakeOpenSearchStore


def opensearch_flow(**parameters):
    node = {
        "name": "os",
        "type": "gwenflow.opensearchQuery",
        "parameters": {"uri": "https://os.test", "index": "docs", **parameters},
    }
    return Flow.from_dict({"name": "opensearch", "nodes": [node]})


def test_opensearch_node_calls_the_store_as_in_v1_5_1(fake_opensearch):
    query = {"match_all": {}}
    assert opensearch_flow(query=query).run() == {"os": [{"id": "1"}]}
    store = fake_opensearch.instances[-1]
    assert store.kwargs == {
        "uri": "https://os.test",
        "index": "docs",
        "use_ssl": True,
        "verify_certs": False,
        "ca_certs": None,
        "timeout": 30,
        "aws": False,
    }
    assert store.calls == [
        (
            "search",
            {"query": query, "aggs": None, "sort": None, "page": 1, "per_page": 5000, "fields": None, "all": False},
        )
    ]
    assert opensearch_flow(operation="count", query=query).run() == {"os": {"count": 3}}
    assert opensearch_flow(operation="get", id="7").run() == {"os": {"id": "7"}}
    assert opensearch_flow(operation="put", documents=[{"id": "1"}]).run() == {"os": {"success": True}}
    assert opensearch_flow(operation="delete", ids=["1"]).run() == {"os": {"success": True}}
    with pytest.raises(FlowNodeError, match="Unknown operation 'drop'"):
        opensearch_flow(operation="drop").run()


def test_opensearch_node_retries_a_search_but_not_a_put(fake_opensearch):
    fake_opensearch.failures = 2
    assert opensearch_flow(query={}).run() == {"os": [{"id": "1"}]}
    assert len(fake_opensearch.instances) == 3  # kept on purpose: 1 attempt in v1.5.1

    fake_opensearch.instances.clear()
    fake_opensearch.failures = 1
    with pytest.raises(FlowNodeError, match="node unavailable"):
        opensearch_flow(operation="put", documents=[{"id": "1"}]).run()
    assert len(fake_opensearch.instances) == 1


class FakeDocument:
    def __init__(self, content, **metadata):
        self.content = content
        self.metadata = metadata


def fake_reader(monkeypatch, module_name, class_name, documents, failures=0):
    """Replace a reader class; returns the list of `read()` calls (positional and keyword arguments)."""
    calls: list = []
    state = {"failures": failures}

    class Reader:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def read(self, *args, **kwargs):
            calls.append((self.kwargs, args, kwargs))
            if state["failures"]:
                state["failures"] -= 1
                raise ConnectionError("temporarily unavailable")
            return documents

    module = types.ModuleType(module_name)
    setattr(module, class_name, Reader)
    monkeypatch.setitem(sys.modules, module_name, module)
    return calls


def reader_flow(type_name, **parameters):
    return Flow.from_dict({"name": "reader", "nodes": [{"name": "read", "type": type_name, "parameters": parameters}]})


def test_website_node_reads_as_in_v1_5_1_and_is_retried(monkeypatch):
    documents = [FakeDocument("page text", url="https://x.test/a", title="A")]
    calls = fake_reader(monkeypatch, "gwenflow.readers.website", "WebsiteReader", documents, failures=2)
    results = reader_flow("gwenflow.websiteReader", url="https://x.test").run()
    assert results == {"read": [{"url": "https://x.test/a", "content": "page text"}]}
    assert calls[0] == ({"max_depth": 1, "max_links": 100000, "delay": False}, ("https://x.test",), {})
    assert len(calls) == 3  # kept on purpose: 1 attempt in v1.5.1


def test_excel_node_reads_as_in_v1_5_1(monkeypatch, tmp_path):
    from pathlib import Path

    documents = [FakeDocument("| a |", sheet="S1", filename="book.xlsx")]
    calls = fake_reader(monkeypatch, "gwenflow.readers.excel", "ExcelReader", documents)
    file = str(tmp_path / "book.xlsx")
    results = reader_flow("gwenflow.excelReader", file=file, sheet_name="S1").run()
    assert results == {"read": [{"content": "| a |", "sheet": "S1", "filename": "book.xlsx"}]}
    assert calls == [({}, (Path(file),), {"sheet_name": "S1", "max_rows": None})]


def test_pdf_node_reads_as_in_v1_5_1_and_is_retried(monkeypatch, tmp_path):
    documents = [FakeDocument("page one", page=1, tables=[["a"]]), FakeDocument("page two", page=2)]
    calls = fake_reader(monkeypatch, "gwenflow.readers.pdf", "PDFReader", documents, failures=1)
    results = reader_flow("gwenflow.pdfReader", file=str(tmp_path / "doc.pdf")).run()
    assert results["read"] == [
        {"page": 1, "content": "page one", "tables": [["a"]]},
        {"page": 2, "content": "page two", "tables": []},
    ]
    assert len(calls) == 2


def test_a_missing_file_is_not_retried(tmp_path):
    with pytest.raises(FlowNodeError, match="No such file"):
        reader_flow("gwenflow.jsonReader", file=str(tmp_path / "missing.json")).run()


def test_python_node_failure_carries_the_stderr_of_the_script(tmp_path):
    (tmp_path / "bad.py").write_text("import sys\nsys.stderr.write('nope')\nsys.exit(3)\n")
    flow = Flow.from_dict(
        {
            "name": "python",
            "nodes": [
                {"name": "py", "type": "gwenflow.Python", "parameters": {"file": "bad.py", "directory": str(tmp_path)}}
            ],
        }
    )
    with pytest.raises(FlowNodeError) as failure:
        flow.run()
    assert "exited with code 3" in str(failure.value) and "nope" in str(failure.value)
    # v1.5.1 raised `subprocess.CalledProcessError` from the handler; it is a `RuntimeError` now.
    assert type(failure.value.__cause__) is RuntimeError


def test_sqlite_and_csv_nodes_as_in_v1_5_1(tmp_path):
    database = str(tmp_path / "db.sqlite")

    def query(name, sql, **extra):
        return {
            "name": name,
            "type": "gwenflow.sqliteQuery",
            "parameters": {"database": database, "query": sql, **extra},
        }

    flow = Flow.from_dict(
        {
            "name": "sqlite",
            "nodes": [
                query("create", "create table t (a integer, b text)"),
                query("insert", "insert into t values (?, ?)", params=[1, "x"]),
                query("select", "select * from t"),
                {"name": "csv", "type": "gwenflow.csvWriter", "parameters": {"file": str(tmp_path / "t.csv")}},
            ],
            "connections": chain("create", "insert", "select", "csv"),
        }
    )
    results = flow.run()
    assert results["select"] == [{"a": 1, "b": "x"}]
    assert (tmp_path / "t.csv").read_text().splitlines() == ["a,b", "1,x"]


# ---------------------------------------------------------------------------
# AutoFlow
# ---------------------------------------------------------------------------

PLAN = [
    {"name": "Biographer", "task": "Write a biography", "tools": [], "depends_on": []},
    {"name": "Summarizer", "task": "List ten points", "tools": None, "depends_on": ["Biographer"]},
    {"name": "Report", "task": "Write the report", "tools": None, "depends_on": ["Biographer", "Summarizer"]},
]


@pytest.fixture
def planner(monkeypatch, provider_keys):
    """An AutoFlow whose llm answers with `PLAN`."""
    from gwenflow import ChatMistral

    class Response:
        content = "```json\n" + json.dumps(PLAN) + "\n```"

    llm = ChatMistral(model="mistral-large-latest")
    monkeypatch.setattr(type(llm), "invoke", lambda self, **kwargs: Response())
    return AutoFlow(llm=llm)


def test_autoflow_plan_writes_the_provider_explicitly(planner):
    flow = planner.plan("A biography")
    for node in flow.nodes:
        assert node.parameters["provider"] == "mistral"
        assert node.parameters["model"] == "mistral-large-latest"
        assert node.parameters["append_inputs"] is True
    # The planned flow, saved and run by `FlowRunner`, uses the client it was planned with.
    llm = agent_llm(flow.node("Report").parameters)
    assert (type(llm).__name__, llm.model) == ("ChatMistral", "mistral-large-latest")


def test_autoflow_run_gives_each_agent_the_outputs_it_depends_on(planner, agent_runs):
    assert planner.run("A biography") == "answer of Report"  # the last agent, as in v1.5.1
    tasks = {run["name"]: run["task"] for run in agent_runs}
    assert tasks["Biographer"] == "Write a biography"
    assert tasks["Summarizer"] == (
        "List ten points\n\nContext from previous steps:\nOutput from Biographer:\nanswer of Biographer"
    )
    assert tasks["Report"] == (
        "Write the report\n\nContext from previous steps:\n"
        "Output from Biographer:\nanswer of Biographer\n\nOutput from Summarizer:\nanswer of Summarizer"
    )
    assert {run["llm"] for run in agent_runs} == {"ChatMistral"}


def test_autoflow_run_works_from_a_running_event_loop(planner, agent_runs):
    assert from_a_running_loop(lambda: planner.run("A biography")) == "answer of Report"


# ---------------------------------------------------------------------------
# Differences with v1.5.1 kept on purpose
# ---------------------------------------------------------------------------


def test_kept_two_nodes_of_the_same_name_are_refused():
    duplicate = {
        "name": "duplicate",
        "nodes": [
            {"name": "x", "type": "compat.Value", "parameters": {"value": 1}},
            {"name": "x", "type": "compat.Value", "parameters": {"value": 2}},
        ],
    }
    with pytest.raises(FlowError, match="Duplicate node names: x"):
        Flow.from_dict(duplicate)


def test_kept_an_invalid_flow_raises_flow_error_and_a_failed_node_flow_node_error(tmp_path):
    path = tmp_path / "bad_target.json"
    path.write_text(
        json.dumps(
            {
                "name": "bad",
                "nodes": [{"name": "a", "type": "compat.Value"}],
                "connections": {"a": [[{"node": "missing"}]]},
            }
        )
    )
    with pytest.raises(FlowError, match="unknown target node 'missing'"):
        FlowRunner(path)

    failing = {
        "name": "failing",
        "nodes": [
            {"name": "first", "type": "compat.Value", "parameters": {"value": 1}},
            {"name": "bad", "type": "compat.Fail"},
        ],
        "connections": chain("first", "bad"),
    }
    with pytest.raises(FlowNodeError) as failure:
        FlowRunner(Flow.from_dict(failing)).run()
    assert failure.value.node == "bad" and str(failure.value) == "Node 'bad' failed: boom"
    assert sum(1 for entry in LOG if entry == ("fail-attempt",)) == 1  # a custom node is not retried


def test_kept_an_unknown_node_type_is_found_before_anything_runs(tmp_path):
    marker = tmp_path / "marker.txt"
    flow = Flow.from_dict(
        {
            "name": "unknown type",
            "nodes": [
                {"name": "mark", "type": "compat.WriteFile", "parameters": {"file": str(marker)}},
                {"name": "ghost", "type": "compat.DoesNotExist"},
            ],
            "connections": chain("mark", "ghost"),
        }
    )
    with pytest.raises(FlowError, match="unknown type 'compat.DoesNotExist'"):
        FlowRunner(flow).run()
    assert not marker.exists()


def test_kept_nodes_run_in_worker_threads():
    FlowRunner(Flow.from_dict(tiny())).run()
    assert [entry[2] for entry in LOG if entry[0] == "value"] == [False]


def test_kept_fan_in_inputs_keep_their_names_and_order():
    flow = Flow.from_dict(
        {
            "name": "fanin",
            "nodes": [
                {"name": "a", "type": "compat.Value", "parameters": {"value": "A"}},
                {"name": "b", "type": "compat.Value", "parameters": {"value": "B"}},
                {"name": "join", "type": "compat.Echo"},
            ],
            "connections": {"a": [[{"node": "join"}]], "b": [[{"node": "join"}]]},
        }
    )
    assert flow.run()["join"] == {"inputs": {"a": "A", "b": "B"}, "order": ["a", "b"]}
