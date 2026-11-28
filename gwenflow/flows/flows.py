import asyncio
import base64
import dataclasses
import datetime
import decimal
import enum
import json
import numbers
import os
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePath
from typing import TYPE_CHECKING, Any

import yaml

from gwenflow.flows.handlers import NODE_TYPE_REGISTRY, is_idempotent, node_paths
from gwenflow.flows.runtime import NodeCancelledError, NodeControl, NodeTimeoutError, run_sync, start_in_thread
from gwenflow.logger import logger
from gwenflow.telemetry import tracer
from gwenflow.utils.durable import in_workflow, new_id

if TYPE_CHECKING:
    import networkx as nx

# networkx costs ~11 MiB of RSS and is only used to plot a flow, so it is
# imported inside the plotting methods rather than at module level.


class FlowError(Exception):
    """A flow that cannot be built or that failed while running."""


class FlowNodeError(FlowError):
    """A node failed; `node` is its name and `__cause__` the original error."""

    def __init__(self, node: str, message: str):
        super().__init__(f"Node '{node}' failed: {message}")
        self.node = node
        self.reason = message


NodeExecutor = Callable[["Node", dict[str, Any] | None], Awaitable[Any]]
NodeCall = Callable[["Node", dict[str, Any] | None], Any]
StatusListener = Callable[[str, str], None]

DEFAULT_IDEMPOTENT_RETRIES = 2
RETRY_INITIAL_INTERVAL = 1.0
RETRY_MAX_INTERVAL = 30.0
STOP_GRACE = 1.0
NON_RETRYABLE_ERRORS: tuple[type[BaseException], ...] = (
    KeyError,
    ValueError,
    TypeError,
    NotImplementedError,
    FileNotFoundError,
    IsADirectoryError,
    NotADirectoryError,
    PermissionError,
)


@dataclass
class Node:
    id: str
    type: str
    name: str = ""
    version: int = 1
    parameters: dict[str, Any] = field(default_factory=dict)
    credentials: dict[str, str] = field(default_factory=dict)
    """Parameter name -> name of the environment variable (or secret) holding its
    value, resolved only when the node runs, so secrets never sit in the flow
    file nor in a durable execution history."""
    position: list[int] = field(default_factory=lambda: [0, 0])
    settings: dict[str, Any] = field(default_factory=dict)
    """Execution settings: `timeout` (seconds) and `retries` (extra attempts)."""


@dataclass(frozen=True)
class ConcurrentPair:
    """Two nodes without a path between them: they may run at the same time."""

    first: str
    second: str
    shared_paths: tuple[str, ...] = ()
    """Files or databases both nodes declare (see `node_paths`): a likely conflict."""


@dataclass
class ConnectionEndpoint:
    node: str
    type: str = "main"
    index: int = 0


@dataclass
class Flow:
    id: str
    name: str
    nodes: list[Node]
    connections: dict[str, list[list[ConnectionEndpoint]]]
    settings: dict[str, Any] = field(default_factory=dict)
    """`node_timeout`, `node_retries`: defaults for every node. `parallel` (default
    true): when false, nodes run one at a time, in `sequential_order()`."""
    durability: Any = field(default=None, repr=False, compare=False)
    """Durable execution backend, e.g. `TemporalDurability()`: nodes become activities
    when the flow runs inside a workflow. Not part of the flow definition."""

    def __post_init__(self) -> None:
        if self.durability is not None:
            self.durability.bind(self)

    # -- loading ---------------------------------------------------------------

    @classmethod
    def from_file(cls, path: str | Path, durability: Any = None) -> "Flow":
        path = Path(path)
        with path.open() as f:
            data = json.load(f) if path.suffix == ".json" else yaml.safe_load(f)
        return cls.from_dict(data, durability=durability)

    @classmethod
    def from_dict(cls, data: dict[str, Any], durability: Any = None) -> "Flow":
        """Build a flow from its YAML/JSON shape (connections keyed by node name)."""
        if not isinstance(data, dict):
            raise FlowError(f"Invalid flow: expected a mapping, got {type(data).__name__}")
        errors: list[str] = []
        nodes = _parse_nodes(data.get("nodes"), errors)
        connections = _parse_connections(data.get("connections"), {n.name: n.id for n in nodes}, errors)
        settings = data.get("settings") or {}
        if not isinstance(settings, dict):
            errors.append("`settings` must be a mapping")
            settings = {}
        for key in ("id", "name"):
            if data.get(key) is not None and not isinstance(data[key], str):
                errors.append(f"`{key}` must be a string")

        flow = cls(
            id=str(data.get("id") or new_id()),
            name=str(data.get("name") or "Untitled"),
            nodes=nodes,
            connections=connections,
            settings=settings,
        )
        errors.extend(flow._problems())
        if errors:
            raise FlowError(_invalid("Invalid flow", errors))
        if durability is not None:
            flow.durability = durability
            durability.bind(flow)
        return flow

    def to_dict(self) -> dict[str, Any]:
        """Inverse of `from_dict`: what a durable execution receives and records."""
        id_to_name = {n.id: n.name for n in self.nodes}
        return {
            "id": self.id,
            "name": self.name,
            "settings": self.settings,
            "nodes": [asdict(n) for n in self.nodes],
            "connections": {
                id_to_name[source]: [
                    [{"node": id_to_name[e.node], "type": e.type, "index": e.index} for e in port] for port in ports
                ]
                for source, ports in self.connections.items()
            },
        }

    # -- running ---------------------------------------------------------------

    async def arun(self, on_status: StatusListener | None = None) -> dict[str, Any]:
        """Run the flow; returns every node's result, keyed by node name.

        Locally, each node runs in a worker thread. Inside a workflow, each node
        runs through the flow's durability (an activity with Temporal).
        `on_status(node_name, status)` is told when a node is running, done, failed,
        cancelled (stopped by another node's failure) or skipped (never started).
        """
        if in_workflow():
            if self.durability is None:
                raise FlowError(
                    f"Flow '{self.name}' runs inside a workflow without durability: "
                    "give it `durability=TemporalDurability()` so its nodes run in activities."
                )
            self.validate()
            return await execute_flow(self, self.durability.execute_node, on_status)
        return await self._arun_locally(on_status=on_status)

    async def _arun_locally(self, call: NodeCall | None = None, on_status: StatusListener | None = None) -> Any:
        """Local run; `call(node, input_data)` replaces the handler call of each attempt."""
        self.validate()

        async def execute_locally(node: Node, input_data: dict[str, Any] | None) -> Any:
            return await run_node_locally(node, input_data, node_execution(node, self), call)

        return await execute_flow(self, execute_locally, on_status)

    def run(self) -> dict[str, Any]:
        """Blocking run. Called from async code, the flow runs in a thread of its own."""
        if in_workflow():
            raise FlowError("Inside a workflow, use `await flow.arun()`.")
        return run_sync(self.arun)

    # -- graph -----------------------------------------------------------------

    def node(self, name: str) -> Node:
        return next(n for n in self.nodes if n.name == name)

    def parents(self) -> dict[str, list[str]]:
        """Node name -> names of the nodes it takes its input from, in declaration order."""
        id_to_name = {n.id: n.name for n in self.nodes}
        parents: dict[str, list[str]] = {n.name: [] for n in self.nodes}
        for source, ports in self.connections.items():
            for port in ports:
                for endpoint in port:
                    target = id_to_name[endpoint.node]
                    if id_to_name[source] not in parents[target]:
                        parents[target].append(id_to_name[source])
        return parents

    def topological_order(self) -> list[str]:
        """Node names, parents first; ties keep declaration order (deterministic)."""
        parents = self.parents()
        done: list[str] = []
        pending = [n.name for n in self.nodes]
        while pending:
            ready = [name for name in pending if all(p in done for p in parents[name])]
            if not ready:
                raise FlowError(f"Flow '{self.name}' has a cycle between: {', '.join(pending)}")
            done.extend(ready)
            pending = [name for name in pending if name not in ready]
        return done

    def sequential_order(self) -> list[str]:
        """Node names in the order v1.5.1 ran them: `networkx.topological_sort` of the graph.

        Used when `settings.parallel` is false. Same generations as
        `topological_order()`, but inside a generation a node comes when its last
        parent releases it (parents in order, their targets in connection order),
        not in declaration order. Computed here: networkx is not imported to run a flow.
        """
        names = [n.name for n in self.nodes]
        id_to_name = {n.id: n.name for n in self.nodes}
        children: dict[str, list[str]] = {name: [] for name in names}
        for source, ports in self.connections.items():
            for port in ports:
                for endpoint in port:
                    target = id_to_name[endpoint.node]
                    if target not in children[id_to_name[source]]:
                        children[id_to_name[source]].append(target)
        waiting = {name: 0 for name in names}
        for targets in children.values():
            for target in targets:
                waiting[target] += 1
        order: list[str] = []
        generation = [name for name in names if waiting[name] == 0]
        while generation:
            order.extend(generation)
            released = []
            for name in generation:
                for child in children[name]:
                    waiting[child] -= 1
                    if waiting[child] == 0:
                        released.append(child)
            generation = released
        if len(order) != len(names):
            raise FlowError(f"Flow '{self.name}' has a cycle between: {', '.join(n for n in names if n not in order)}")
        return order

    def concurrent_pairs(self) -> list[tuple[str, str]]:
        """Pairs of nodes without a path between them, in declaration order.

        With `settings.parallel` left to true, the two nodes of a pair may run at
        the same time: check that they do not write the same file, table or
        resource. `concurrent_report()` adds the paths both declare.
        """
        parents = self.parents()
        ancestors: dict[str, set[str]] = {}
        for name in self.topological_order():
            ancestors[name] = set(parents[name]).union(*(ancestors[p] for p in parents[name]))
        names = [n.name for n in self.nodes]
        return [
            (a, b)
            for i, a in enumerate(names)
            for b in names[i + 1 :]
            if a not in ancestors[b] and b not in ancestors[a]
        ]

    def concurrent_report(self) -> list[ConcurrentPair]:
        """`concurrent_pairs()`, each with the paths (files, SQLite databases) both nodes declare."""
        declared = {n.name: {str(Path(p).resolve()): p for p in node_paths(n.type, n.parameters)} for n in self.nodes}
        return [
            ConcurrentPair(a, b, tuple(path for key, path in declared[a].items() if key in declared[b]))
            for a, b in self.concurrent_pairs()
        ]

    def sinks(self) -> list[str]:
        """Nodes nobody consumes: the outputs of the flow."""
        consumed = {p for ps in self.parents().values() for p in ps}
        return [n.name for n in self.nodes if n.name not in consumed]

    def validate(self, node_types: Mapping[str, Any] | None = None) -> None:
        problems = self._problems(NODE_TYPE_REGISTRY if node_types is None else node_types)
        if problems:
            raise FlowError(_invalid(f"Invalid flow '{self.name}'", problems))

    def _problems(self, node_types: Mapping[str, Any] | None = None) -> list[str]:
        problems = []
        for label, values in (("names", [n.name for n in self.nodes]), ("ids", [n.id for n in self.nodes])):
            duplicates = sorted({v for v in values if values.count(v) > 1})
            if duplicates:
                problems.append(f"Duplicate node {label}: {', '.join(duplicates)}")
        known = {n.id for n in self.nodes}
        endpoints = list(self.connections) + [
            e.node for ports in self.connections.values() for port in ports for e in port
        ]
        problems.extend(
            f"Connection references unknown node id '{i}'" for i in dict.fromkeys(endpoints) if i not in known
        )
        for node in self.nodes:
            problems.extend(f"node '{node.name}': {p}" for p in _settings_problems(node.settings, "timeout", "retries"))
            if node_types is not None and node.type not in node_types:
                problems.append(f"node '{node.name}' has an unknown type '{node.type}'")
        problems.extend(
            f"flow settings: {p}" for p in _settings_problems(self.settings, "node_timeout", "node_retries")
        )
        if not isinstance(self.settings.get("parallel", True), bool):
            problems.append("flow settings: `parallel` must be true or false")
        if not problems:
            try:
                self.topological_order()
            except FlowError as e:
                problems.append(str(e))
        return problems


def _invalid(title: str, errors: list[str]) -> str:
    return f"{title}:\n" + "\n".join(f"  - {e}" for e in errors)


def _parse_nodes(raw: Any, errors: list[str]) -> list[Node]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        errors.append("`nodes` must be a list")
        return []
    nodes = []
    for index, item in enumerate(raw, 1):
        before = len(errors)
        label = f"node #{index}"
        if not isinstance(item, dict):
            errors.append(f"{label} must be a mapping")
            continue
        name, type_ = item.get("name"), item.get("type")
        if isinstance(name, str) and name:
            label = f"node '{name}'"
        else:
            errors.append(f"{label} needs a non-empty `name`")
        if not (isinstance(type_, str) and type_):
            errors.append(f"{label} needs a non-empty `type`")
        for key in ("parameters", "credentials", "settings"):
            if item.get(key) is not None and not isinstance(item[key], dict):
                errors.append(f"{label}: `{key}` must be a mapping")
        credentials = item.get("credentials")
        if isinstance(credentials, dict):
            # v1.5.1 ignored the whole block, and flows exported from other tools carry
            # one (`{"api": {"id": "1", "name": "my key"}}`): what does not name an
            # environment variable is left out, not refused.
            ignored = [key for key, value in credentials.items() if not (isinstance(value, str) and value)]
            if ignored:
                logger.warning(
                    f"Flow {label}: credentials ignored ({', '.join(map(str, ignored))}): "
                    "a credential must name an environment variable."
                )
            credentials = {key: value for key, value in credentials.items() if key not in ignored}
        if item.get("id") is not None and not isinstance(item["id"], str):
            errors.append(f"{label}: `id` must be a string")
        if not isinstance(item.get("version", 1), int):
            errors.append(f"{label}: `version` must be an integer")
        if not isinstance(item.get("position", []), list):
            errors.append(f"{label}: `position` must be a list")
        if len(errors) > before:
            continue
        nodes.append(
            Node(
                id=item.get("id") or new_id(),
                type=type_,
                name=name,
                version=item.get("version", 1),
                position=item.get("position", [0, 0]),
                parameters=item.get("parameters") or {},
                credentials=credentials or {},
                settings=item.get("settings") or {},
            )
        )
    return nodes


def _parse_connections(
    raw: Any, name_to_id: dict[str, str], errors: list[str]
) -> dict[str, list[list[ConnectionEndpoint]]]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        errors.append("`connections` must be a mapping of source node name to output ports")
        return {}
    connections: dict[str, list[list[ConnectionEndpoint]]] = {}
    for source_name, ports in raw.items():
        source_id = name_to_id.get(source_name)
        if source_id is None:
            errors.append(f"Connection references unknown source node '{source_name}'")
            continue
        if not isinstance(ports, list) or not all(isinstance(port, list) for port in ports):
            errors.append(f"connections of '{source_name}' must be a list of output ports, each a list of targets")
            continue
        connections[source_id] = []
        for port in ports:
            targets = []
            for target in port:
                if not isinstance(target, dict) or not isinstance(target.get("node"), str):
                    errors.append(f"a connection of '{source_name}' needs a target `node`")
                    continue
                target_id = name_to_id.get(target["node"])
                if target_id is None:
                    errors.append(f"Connection references unknown target node '{target['node']}'")
                    continue
                targets.append(
                    ConnectionEndpoint(node=target_id, type=target.get("type", "main"), index=target.get("index", 0))
                )
            connections[source_id].append(targets)
    return connections


def _settings_problems(settings: dict[str, Any], timeout_key: str, retries_key: str) -> list[str]:
    problems = []
    timeout = settings.get(timeout_key)
    if timeout is not None and (isinstance(timeout, bool) or not isinstance(timeout, int | float) or timeout <= 0):
        problems.append(f"`{timeout_key}` must be a positive number of seconds")
    retries = settings.get(retries_key)
    if retries is not None and (isinstance(retries, bool) or not isinstance(retries, int) or retries < 0):
        problems.append(f"`{retries_key}` must be a whole number, 0 or more")
    return problems


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


async def execute_flow(
    flow: Flow, execute_node: NodeExecutor, on_status: StatusListener | None = None
) -> dict[str, Any]:
    """Run every node as soon as all of its parents have finished.

    Independent branches run concurrently. `execute_node` does the work: a
    thread locally, an activity in a Temporal workflow. Only asyncio tasks are
    used here, so this function is valid workflow code. The first failure
    cancels the nodes still running and is raised as `FlowNodeError`.

    With `settings.parallel` set to false, nodes run one at a time instead, in
    `flow.sequential_order()`, as before v1.5.2.
    """
    parents = flow.parents()
    tasks: dict[str, asyncio.Task] = {}

    if flow.settings.get("parallel", True) is False:
        return await _execute_one_by_one(flow, execute_node, on_status)

    async def run(name: str) -> Any:
        try:
            inputs = [await tasks[p] for p in parents[name]]
        except BaseException:
            notify(name, "skipped")
            raise
        input_data = dict(zip(parents[name], inputs, strict=True)) if parents[name] else None
        notify(name, "running")
        try:
            result = await execute_node(flow.node(name), input_data)
        except asyncio.CancelledError:
            notify(name, "cancelled")
            raise
        except FlowNodeError:
            notify(name, "failed")
            raise
        except Exception as e:
            notify(name, "failed")
            raise FlowNodeError(name, describe_error(e)) from e
        notify(name, "done")
        return result

    def notify(name: str, status: str) -> None:
        if on_status is not None:
            on_status(name, status)

    for name in flow.topological_order():
        tasks[name] = asyncio.ensure_future(run(name))
    try:
        await asyncio.gather(*tasks.values())
    except BaseException:
        for task in tasks.values():
            task.cancel()
        await asyncio.gather(*tasks.values(), return_exceptions=True)
        raise
    return {name: task.result() for name, task in tasks.items()}


async def _execute_one_by_one(
    flow: Flow, execute_node: NodeExecutor, on_status: StatusListener | None = None
) -> dict[str, Any]:
    parents = flow.parents()
    order = flow.sequential_order()
    results: dict[str, Any] = {}

    def notify(name: str, status: str) -> None:
        if on_status is not None:
            on_status(name, status)

    for position, name in enumerate(order):
        input_data = {p: results[p] for p in parents[name]} if parents[name] else None
        notify(name, "running")
        try:
            results[name] = await execute_node(flow.node(name), input_data)
        except BaseException as e:
            notify(name, "cancelled" if isinstance(e, asyncio.CancelledError) else "failed")
            for skipped in order[position + 1 :]:
                notify(skipped, "skipped")
            if isinstance(e, asyncio.CancelledError | FlowNodeError) or not isinstance(e, Exception):
                raise
            raise FlowNodeError(name, describe_error(e)) from e
        notify(name, "done")
    return results


@dataclass(frozen=True)
class NodeExecution:
    timeout: float | None = None
    retries: int = 0

    @property
    def attempts(self) -> int:
        return 1 + self.retries


def node_execution(node: Node, flow: Flow | None = None) -> NodeExecution:
    flow_settings = flow.settings if flow is not None else {}
    timeout = _first_set(node.settings.get("timeout"), flow_settings.get("node_timeout"))
    retries = _first_set(node.settings.get("retries"), flow_settings.get("node_retries"))
    if retries is None:
        retries = DEFAULT_IDEMPOTENT_RETRIES if is_idempotent(node.type, node.parameters) else 0
    return NodeExecution(timeout=float(timeout) if timeout is not None else None, retries=int(retries))


def _first_set(*values: Any) -> Any:
    return next((v for v in values if v is not None), None)


def retry_delay(attempt: int) -> float:
    return min(RETRY_INITIAL_INTERVAL * 2 ** (attempt - 1), RETRY_MAX_INTERVAL)


def is_retryable(error: BaseException) -> bool:
    if isinstance(error, NodeTimeoutError):
        return not error.still_running
    return not isinstance(error, (FlowNodeError, NodeCancelledError, *NON_RETRYABLE_ERRORS))


def describe_error(error: BaseException) -> str:
    error = getattr(error, "cause", None) or error
    text = str(error)
    if isinstance(error, KeyError):
        return f"KeyError: {text}"
    return text or type(error).__name__


def resolve_credentials(node: Node) -> dict[str, Any]:
    """The node's parameters with its credentials filled in from the environment."""
    parameters = dict(node.parameters)
    for param, variable in node.credentials.items():
        value = os.environ.get(variable)
        if value is None:
            raise FlowNodeError(node.name, f"credential '{param}' needs the environment variable {variable}")
        _set_parameter(parameters, param, value, node.name)
    return parameters


def _set_parameter(parameters: dict[str, Any], path: str, value: str, node_name: str) -> None:
    *parents, leaf = path.split(".")
    target = parameters
    for key in parents:
        child = target.get(key, {})
        if not isinstance(child, dict):
            raise FlowNodeError(node_name, f"credential '{path}': '{key}' is not a mapping")
        child = dict(child)
        target[key] = child
        target = child
    target[leaf] = value


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | bool | int | float):
        return value
    if isinstance(value, dict):
        return {_json_key(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [to_jsonable(v) for v in value]
    if isinstance(value, datetime.datetime | datetime.date | datetime.time):
        return value.isoformat()
    if isinstance(value, datetime.timedelta):
        return value.total_seconds()
    if isinstance(value, decimal.Decimal | uuid.UUID | PurePath):
        return str(value)
    if isinstance(value, bytes | bytearray | memoryview):
        return base64.b64encode(bytes(value)).decode("ascii")
    if isinstance(value, enum.Enum):
        return to_jsonable(value.value)
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return float(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(dataclasses.asdict(value))
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return to_jsonable(model_dump(mode="json"))
    raise TypeError(f"a {type(value).__name__} is not JSON serializable")


def _json_key(key: Any) -> str:
    if isinstance(key, str):
        return key
    if key is None or isinstance(key, bool | int | float):
        return json.dumps(key)
    return str(to_jsonable(key))


def call_handler(node_type: str, parameters: dict[str, Any], input_data: dict[str, Any] | None) -> Any:
    """The handler's result, as it is: locally, the next node receives the same object."""
    handler = NODE_TYPE_REGISTRY.get(node_type)
    if handler is None:
        raise NotImplementedError(f"Node type '{node_type}' is not supported.")
    return handler(parameters, input_data)


def jsonable_result(node_type: str, result: Any) -> Any:
    """A node result as JSON data, for a durable execution: it is recorded in the history."""
    try:
        return to_jsonable(result)
    except TypeError as e:
        raise TypeError(
            f"node type '{node_type}' returned a value that is not JSON serializable ({e}): "
            "return dicts, lists, strings, numbers, booleans or None"
        ) from e


async def run_in_thread(fn: Callable[..., Any], *args: Any, timeout: float | None = None) -> Any:
    control = NodeControl(timeout)
    future = start_in_thread(control, fn, *args)
    try:
        done, _ = await asyncio.wait({future}, timeout=timeout)
    except asyncio.CancelledError:
        control.stop()
        raise
    if done:
        return future.result()
    control.stop()
    done, _ = await asyncio.wait({future}, timeout=STOP_GRACE)
    if not done:
        raise NodeTimeoutError(timeout, still_running=True)
    raise NodeTimeoutError(timeout) from future.exception()


async def run_node_locally(
    node: Node,
    input_data: dict[str, Any] | None,
    execution: NodeExecution | None = None,
    call: NodeCall | None = None,
) -> Any:
    """Local executor: a worker thread, with the node's timeout and retries.

    `call(node, input_data)` does one attempt in place of the handler call
    (`FlowRunner._execute_node`); it runs in the worker thread too.
    """
    execution = execution or node_execution(node)
    if call is None:
        parameters = resolve_credentials(node)

        def call(node: Node, input_data: dict[str, Any] | None) -> Any:
            return call_handler(node.type, parameters, input_data)

    for attempt in range(1, execution.attempts + 1):
        logger.info(f"Running node: {node.name} [{node.type}]" + (f" (attempt {attempt})" if attempt > 1 else ""))
        try:
            return await run_in_thread(call, node, input_data, timeout=execution.timeout)
        except Exception as e:
            if attempt == execution.attempts or not is_retryable(e):
                raise
            delay = retry_delay(attempt)
            logger.warning(f"Node '{node.name}' failed ({describe_error(e)}), retrying in {delay:g}s.")
            await asyncio.sleep(delay)


class FlowRunner:
    def __init__(self, flow_path: "str | Path | Flow"):
        """`flow_path`: a YAML/JSON flow file, or an already built `Flow`."""
        self.flow = flow_path if isinstance(flow_path, Flow) else Flow.from_file(Path(flow_path))
        self._graph: "nx.DiGraph | None" = None

    @property
    def graph(self) -> "nx.DiGraph":
        """The flow as a networkx graph (node ids, edge ports), built on first use."""
        if self._graph is None:
            self._graph = self._build_graph()
        return self._graph

    def _execute_node(self, node: Node, input_data: dict[str, Any] | None) -> Any:
        """Run one attempt of one node, and return its result.

        Override it to wrap or replace what a node does (audit, cache, test
        double): every local run of this runner goes through it. It is called
        in the node's worker thread, once per attempt, and the node's timeout and
        retries apply around it. Inside a Temporal workflow nodes are activities,
        run by the worker: this method is not called.
        """
        return call_handler(node.type, resolve_credentials(node), input_data)

    async def arun(self) -> dict[str, Any]:
        """Run the flow; returns every node's result, keyed by node name."""
        return await self._arun(echo=False)

    async def _arun(self, echo: bool) -> dict[str, Any]:
        with tracer.span("Flow Run"):
            if in_workflow():
                results = await self.flow.arun()
            else:
                results = await self.flow._arun_locally(self._execute_node)
        if echo and results:
            # As v1.5.1 did: the result of the last node, in the order it ran them.
            print(results[self.flow.sequential_order()[-1]])  # noqa: T201
        logger.info(f"Flow '{self.flow.name}' completed.")
        return results

    def run(self) -> dict[str, Any]:
        """Blocking run: prints the last node's result, and returns every node's result.

        Called from async code (a running event loop), the flow runs in a thread
        of its own. Use `await runner.arun()` there to run it on the loop, without
        the print.
        """
        if in_workflow():
            raise FlowError("Inside a workflow, use `await runner.arun()`.")
        return run_sync(lambda: self._arun(echo=True))

    # -- plotting --------------------------------------------------------------

    def _build_graph(self) -> "nx.DiGraph":
        import networkx as nx

        graph = nx.DiGraph()
        for node in self.flow.nodes:
            graph.add_node(node.id, data=node)
        for source_id, port_list in self.flow.connections.items():
            for output_index, targets in enumerate(port_list):
                for endpoint in targets:
                    graph.add_edge(
                        source_id,
                        endpoint.node,
                        output_index=output_index,
                        input_type=endpoint.type,
                        input_index=endpoint.index,
                    )
        return graph

    def plot(self, filename="pipeline_graph"):
        import networkx as nx

        p = nx.drawing.nx_pydot.to_pydot(self.graph)
        for node in p.get_nodes():
            node.set_shape("box")
            node.set_style("filled")
            node.set_fillcolor("#eeeeee")
        p.write_png(f"{filename}.png")

    def plot_interactive(self):
        from pyvis.network import Network

        graph = self.graph
        net = Network(notebook=False, directed=True, bgcolor="#222222", font_color="white")

        node_styles = {
            "data": {"color": {"background": "#4da6ff", "border": "#0059b3"}, "shape": "ellipse", "size": 20},
            "python": {"color": {"background": "#5cd65c", "border": "#2d862d"}, "shape": "box", "size": 25},
            "agent": {"color": {"background": "#ffad33", "border": "#b36b00"}, "shape": "dot", "size": 30},
        }

        for node_id in graph.nodes:
            node_obj = graph.nodes[node_id]["data"]
            style = node_styles.get(
                node_obj.type.lower(),
                {"color": {"background": "#97c2fc", "border": "#2b7ce9"}, "shape": "dot", "size": 25},
            )
            net.add_node(
                node_id,
                label=f"{node_obj.name}\n{node_obj.type}",
                shape=style["shape"],
                color=style["color"],
                size=style["size"],
                font={"color": "white", "size": 14},
            )

        for source, target, data in graph.edges(data=True):
            net.add_edge(source, target, title=data.get("input_type"), color="#848484")

        net.show("pipeline.html", notebook=False)
