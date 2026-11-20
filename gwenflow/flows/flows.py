import asyncio
import json
import os
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from gwenflow.flows.handlers import NODE_TYPE_REGISTRY
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


NodeExecutor = Callable[["Node", dict[str, Any] | None], Awaitable[Any]]
StatusListener = Callable[[str, str], None]


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
        nodes = [
            Node(
                id=n.get("id") or new_id(),
                type=n["type"],
                name=n["name"],
                version=n.get("version", 1),
                position=n.get("position", [0, 0]),
                parameters=n.get("parameters") or {},
                credentials=n.get("credentials") or {},
                settings=n.get("settings") or {},
            )
            for n in data.get("nodes", [])
        ]
        errors = []
        names = [n.name for n in nodes]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            errors.append(f"Duplicate node names: {', '.join(duplicates)}")

        name_to_id = {n.name: n.id for n in nodes}
        connections: dict[str, list[list[ConnectionEndpoint]]] = {}
        for source_name, port_list in (data.get("connections") or {}).items():
            source_id = name_to_id.get(source_name)
            if source_id is None:
                errors.append(f"Connection references unknown source node '{source_name}'")
                continue
            connections[source_id] = []
            for port_targets in port_list:
                targets = []
                for t in port_targets:
                    target_id = name_to_id.get(t["node"])
                    if target_id is None:
                        errors.append(f"Connection references unknown target node '{t['node']}'")
                    else:
                        targets.append(
                            ConnectionEndpoint(node=target_id, type=t.get("type", "main"), index=t.get("index", 0))
                        )
                connections[source_id].append(targets)

        if errors:
            raise FlowError("Invalid flow:\n" + "\n".join(f"  - {e}" for e in errors))

        flow = cls(
            id=data.get("id") or new_id(),
            name=data.get("name", "Untitled"),
            nodes=nodes,
            connections=connections,
            settings=data.get("settings", {}),
        )
        flow.topological_order()  # fails early on a cycle
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
        `on_status(node_name, status)` is told when a node is running, done or failed.
        """
        if in_workflow():
            if self.durability is None:
                raise FlowError(
                    f"Flow '{self.name}' runs inside a workflow without durability: "
                    "give it `durability=TemporalDurability()` so its nodes run in activities."
                )
            return await execute_flow(self, self.durability.execute_node, on_status)
        return await execute_flow(self, run_node_locally, on_status)

    def run(self) -> dict[str, Any]:
        if in_workflow():
            raise FlowError("Inside a workflow, use `await flow.arun()`.")
        return asyncio.run(self.arun())

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

    def sinks(self) -> list[str]:
        """Nodes nobody consumes: the outputs of the flow."""
        consumed = {p for ps in self.parents().values() for p in ps}
        return [n.name for n in self.nodes if n.name not in consumed]


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
    """
    parents = flow.parents()
    tasks: dict[str, asyncio.Task] = {}

    async def run(name: str) -> Any:
        inputs = [await tasks[p] for p in parents[name]]
        input_data = dict(zip(parents[name], inputs, strict=True)) if parents[name] else None
        notify(name, "running")
        try:
            result = await execute_node(flow.node(name), input_data)
        except asyncio.CancelledError:
            raise
        except FlowNodeError:
            notify(name, "failed")
            raise
        except Exception as e:
            notify(name, "failed")
            raise FlowNodeError(name, str(getattr(e, "cause", None) or e)) from e
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


def resolve_credentials(node: Node) -> dict[str, Any]:
    """The node's parameters with its credentials filled in from the environment."""
    parameters = dict(node.parameters)
    for param, variable in node.credentials.items():
        value = os.environ.get(variable)
        if value is None:
            raise FlowNodeError(node.name, f"credential '{param}' needs the environment variable {variable}")
        parameters[param] = value
    return parameters


def call_handler(node_type: str, parameters: dict[str, Any], input_data: dict[str, Any] | None) -> Any:
    handler = NODE_TYPE_REGISTRY.get(node_type)
    if handler is None:
        raise NotImplementedError(f"Node type '{node_type}' is not supported.")
    return handler(parameters, input_data)


async def run_node_locally(node: Node, input_data: dict[str, Any] | None) -> Any:
    """Local executor: a worker thread, with the node's timeout and retries."""
    attempts = 1 + int(node.settings.get("retries", 0))
    timeout = node.settings.get("timeout")
    for attempt in range(1, attempts + 1):
        logger.info(f"Running node: {node.name} [{node.type}]" + (f" (attempt {attempt})" if attempt > 1 else ""))
        try:
            call = asyncio.to_thread(call_handler, node.type, resolve_credentials(node), input_data)
            return await asyncio.wait_for(call, timeout) if timeout else await call
        except (NotImplementedError, FlowNodeError):
            raise
        except Exception as e:
            if attempt == attempts:
                raise
            logger.warning(f"Node '{node.name}' failed ({e}), retrying.")
            await asyncio.sleep(min(2 ** (attempt - 1), 10))


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

    async def arun(self) -> dict[str, Any]:
        """Run the flow; returns every node's result, keyed by node name."""
        with tracer.span("Flow Run"):
            results = await self.flow.arun()
        logger.info(f"Flow '{self.flow.name}' completed.")
        return results

    def run(self) -> dict[str, Any]:
        return asyncio.run(self.arun())

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
