import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx
import yaml

from gwenflow.flows.handlers import NODE_TYPE_REGISTRY
from gwenflow.logger import logger


@dataclass
class Node:
    id: str
    type: str
    name: str = ""
    version: int = 1
    parameters: dict[str, Any] = field(default_factory=dict)
    credentials: dict[str, Any] = field(default_factory=dict)
    position: list[int] = field(default_factory=lambda: [0, 0])


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

    @classmethod
    def from_file(cls, path: Path) -> "Flow":
        with path.open() as f:
            if path.suffix == ".json":
                data = json.load(f)
            else:
                data = yaml.safe_load(f)

        nodes = [
            Node(
                id=n.get("id") or str(uuid.uuid4()),
                type=n["type"],
                name=n["name"],
                version=n.get("version", 1),
                position=n.get("position", [0, 0]),
                parameters=n.get("parameters") or {},
            )
            for n in data.get("nodes", [])
        ]

        name_to_id = {n.name: n.id for n in nodes}
        raw_connections = data.get("connections", {})
        connections: dict[str, list[list[ConnectionEndpoint]]] = {}
        errors = []

        for source_name, port_list in raw_connections.items():
            source_id = name_to_id.get(source_name)
            if source_id is None:
                errors.append(f"Connection references unknown source node '{source_name}'")
                continue
            connections[source_id] = []
            for port_targets in port_list:
                targets = []
                for t in port_targets:
                    target_name = t["node"]
                    target_id = name_to_id.get(target_name)
                    if target_id is None:
                        errors.append(f"Connection references unknown target node '{target_name}'")
                    else:
                        targets.append(
                            ConnectionEndpoint(
                                node=target_id,
                                type=t.get("type", "main"),
                                index=t.get("index", 0),
                            )
                        )
                connections[source_id].append(targets)

        if errors:
            raise ValueError("Invalid flow YAML:\n" + "\n".join(f"  - {e}" for e in errors))

        return cls(
            id=data.get("id") or str(uuid.uuid4()),
            name=data.get("name", "Untitled"),
            nodes=nodes,
            connections=connections,
            settings=data.get("settings", {}),
        )


class FlowRunner:
    def __init__(self, flow_path: str | Path):
        self.flow = Flow.from_file(Path(flow_path))
        self.graph = self._build_graph()

    def _build_graph(self) -> nx.DiGraph:
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

    def _execute_node(self, node: Node, input_data: dict[str, str] | None) -> Any:
        logger.info(f"Running Node: {node.name} [{node.type}] ")
        handler = NODE_TYPE_REGISTRY.get(node.type)
        if handler is None:
            raise NotImplementedError(f"Node type '{node.type}' is not supported.")
        return handler(node.parameters, input_data)

    def plot(self, filename="pipeline_graph"):
        p = nx.drawing.nx_pydot.to_pydot(self.graph)
        for node in p.get_nodes():
            node.set_shape("box")
            node.set_style("filled")
            node.set_fillcolor("#eeeeee")
        p.write_png(f"{filename}.png")

    def plot_interactive(self):
        from pyvis.network import Network

        net = Network(notebook=False, directed=True, bgcolor="#222222", font_color="white")

        node_styles = {
            "data": {"color": {"background": "#4da6ff", "border": "#0059b3"}, "shape": "ellipse", "size": 20},
            "python": {"color": {"background": "#5cd65c", "border": "#2d862d"}, "shape": "box", "size": 25},
            "agent": {"color": {"background": "#ffad33", "border": "#b36b00"}, "shape": "dot", "size": 30},
        }

        for node_id in self.graph.nodes:
            node_obj = self.graph.nodes[node_id]["data"]
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

        for source, target, data in self.graph.edges(data=True):
            net.add_edge(source, target, title=data.get("input_type"), color="#848484")

        net.show("pipeline.html", notebook=False)

    def run(self) -> None:
        order = list(nx.topological_sort(self.graph))
        results: dict[str, Any] = {}
        for node_id in order:
            node: Node = self.graph.nodes[node_id]["data"]
            parents = list(self.graph.predecessors(node_id))
            input_data = {self.graph.nodes[p]["data"].name: results[p] for p in parents} if parents else None
            try:
                results[node_id] = self._execute_node(node, input_data)
            except Exception as e:
                logger.warning(f"Flow '{node.name}' stopped at : {e}", err=True)
                raise SystemExit(1)
        print(results[node_id])
        logger.info("Flow completed.")
