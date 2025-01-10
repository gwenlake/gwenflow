import json
from collections import defaultdict, deque
from pydantic import BaseModel, Field, field_validator
from gwenflow.flows import Flow
from gwenflow.tools import WikipediaTool, WebsiteTool, DuckDuckGoSearchTool
from jinja2 import Environment, FileSystemLoader
from markupsafe import Markup

HTML_TEMPLATE = """
<div>
  <div class="title-box">
    <i class="fas fa-robot"></i>&nbsp;&nbsp;{name}
  </div>
  <div class="box">
    <p>{task}</p>
  </div>
</div>
"""

def escapejs(value):
    """
    Escapes special characters for JavaScript.
    """
    if value is None:
        return ''
    escaped = (
        value.replace('\\', '\\\\')
             .replace("'", "\\'")
             .replace('"', '\\"')
             .replace('\n', '\\n')
             .replace('\r', '\\r')
             .replace('\t', '\\t')
    )
    return Markup(escaped)

class Node(BaseModel):
    id: str
    name: str
    context_vars: list[str] = Field(default_factory=list)
    role: str
    inputs: dict = Field(default_factory=lambda: {"input_1": {"connections": []}})
    outputs: dict = Field(default_factory=lambda: {"output_1": {"connections": []}})
    pos_x: int = 0
    pos_y: int = 0

    @field_validator("id", "name", "role", mode="before")
    def non_empty_string(cls, value):
        if not value.strip():
            raise ValueError("Field cannot be an empty string.")
        return value

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "context_vars": self.context_vars,
            "class": "agent",
            "html": HTML_TEMPLATE.format(name=self.name, task=self.role),
            "typenode": False,
            "data": {},
            "inputs": self.inputs,
            "outputs": self.outputs,
            "pos_x": self.pos_x,
            "pos_y": self.pos_y,
        }

class DrawFlow(BaseModel):
    flow: Flow
    nodes: dict[str, Node] = Field(default_factory=dict)

    def _initialize_nodes(self):
        """Creates Node objects from the flow agents."""
        self.nodes = {
            str(agent.id): Node(
                id=str(agent.id),
                name=agent.name,
                context_vars=agent.context_vars,
                role=agent.role
            )
            for agent in self.flow.__dict__["agents"]
        }

    def _find_longest_path_to_sources(self, start_node):
        """Finds the longest path to sources in a graph. Based on BFS Algorithm."""
        adj_list = defaultdict(list)
        in_degree = {node_id: 0 for node_id in self.nodes}

        for node in self.nodes.values():
            if not node.outputs:
                continue
            for conn in node.outputs["output_1"]["connections"]:
                adj_list[node.id].append(conn["node"])
                adj_list[conn["node"]].append(node.id)
                in_degree[conn["node"]] += 1

        sources = [node_id for node_id, degree in in_degree.items() if degree == 0]
        distances = {node_id: float('inf') for node_id in self.nodes}
        distances[start_node] = 0

        queue = deque([start_node])
        while queue:
            current = queue.popleft()
            for neighbor in adj_list[current]:
                if distances[neighbor] > distances[current] + 1:
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)

        return min((distances[source] for source in sources), default=float('inf'))

    def _assign_io_connections(self):
        """Assigns input and output connections between nodes based on context variables."""
        for node in self.nodes.values():
            for other_node in self.nodes.values():
                if node.id == other_node.id:
                    continue
                if node.name in other_node.context_vars:
                    node.outputs["output_1"]["connections"].append({
                        "node": other_node.id,
                        "output": "input_1"
                    })
                    other_node.inputs["input_1"]["connections"].append({
                        "node": node.id,
                        "input": "output_1"
                    })
        for node in self.nodes.values():
            if not node.inputs:
                continue
            if len(node.inputs["input_1"]["connections"]) == 0:
                node.inputs = {}
            if not node.outputs:
                continue
            if len(node.outputs["output_1"]["connections"]) == 0:
                node.outputs = {}

    def _assign_positions(self, separation_x=400, separation_y=200, initial_x=-350, initial_y=-150):
        """Calculates and assigns positions to nodes."""
        columns = {node_id: self._find_longest_path_to_sources(node_id) + 1 for node_id in self.nodes}
        rows = defaultdict(int)
        columns_inv = defaultdict(list)

        for node_id, col in columns.items():
            columns_inv[col].append(node_id)

        for col, nodes in columns_inv.items():
            for index, node_id in enumerate(nodes):
                rows[node_id] = index + 1

        for node_id, node in self.nodes.items():
            node.pos_x = initial_x + columns[node_id] * separation_x
            node.pos_y = initial_y + rows[node_id] * separation_y

    def generate_drawflow(self):
        """Generates the drawflow JSON structure."""
        self._assign_io_connections()
        self._assign_positions()
        return {
            "drawflow": {
                "Home": {
                    "data": {node_id: node.to_dict() for node_id, node in self.nodes.items()}
                }
            }
        }

    @classmethod
    def from_yaml(cls, yaml_path, tools):
        """Creates a DrawFlow instance from a YAML configuration."""
        flow = Flow().from_yaml(yaml_path, tools=tools)
        instance = cls(flow=flow)
        instance._initialize_nodes()
        return instance
    
    @classmethod
    def from_flow(cls, flow):
        """Creates a DrawFlow instance from a Flow instance."""
        instance = cls(flow=flow)
        instance._initialize_nodes()
        return instance

    def save_to_json(self, json_path):
        """Saves the generated drawflow to a JSON file."""
        drawflow_json = self.generate_drawflow()
        with open(json_path, "w", encoding="utf-8") as json_file:
            json.dump(drawflow_json, json_file, indent=4, ensure_ascii=False)

    def generate_html(self, output_file):
        """Generates an HTML file using a Jinja2 template."""
        template_file = "index.html"
        drawflow_json = self.generate_drawflow()
        env = Environment(loader=FileSystemLoader(""))
        env.filters['escapejs'] = escapejs

        template = env.get_template(template_file)

        rendered_html = template.render({
            "drawflow_json": json.dumps(drawflow_json)
        })

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(rendered_html)

        print(f"HTML generated successfully at {output_file}")