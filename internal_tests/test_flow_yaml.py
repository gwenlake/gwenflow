import context

import json

from gwenflow import Flow, Tool
from gwenflow import ChatOpenAI, Agent, Tool
from gwenflow.tools import WikipediaTool, WebsiteTool, PDFTool, DuckDuckGoSearchTool, DuckDuckGoNewsTool
from gwenflow.utils import set_log_level_to_debug
from collections import defaultdict, deque

set_log_level_to_debug()

HTML = """<p>agent : {name}
task : {task}</p>
"""

def find_longest_path_to_sources(graph, start_node):
    from collections import defaultdict, deque

    # Construire liste d'adjacence et degrés entrants
    adj_list = defaultdict(list)
    in_degree = {node: 0 for node in graph}
    for node, data in graph.items():
        for conn in data["outputs"]["output"]["connections"]:
            adj_list[node].append(conn["node"])
            in_degree[conn["node"]] += 1

    # Trouver les nœuds sources (sans degré entrant)
    sources = [node for node, degree in in_degree.items() if degree == 0]

    # Initialiser les longueurs maximales
    longest_path = {node: float('-inf') for node in graph}
    longest_path[start_node] = 0

    # Effectuer un tri topologique inversé
    stack = deque([start_node])
    while stack:
        current = stack.pop()
        for neighbor in adj_list[current]:
            if longest_path[current] + 1 > longest_path[neighbor]:
                longest_path[neighbor] = longest_path[current] + 1
            stack.append(neighbor)

    # Trouver la longueur maximale parmi les nœuds sources
    return max((longest_path[source] for source in sources if longest_path[source] != float('-inf')), default=0)


def get_io(flow: dict):
    """Get inputs and ouputs of each agent to match with drawflow structure.

    Args:
        flow (dict): data format given to drawflow

    Returns:
        dict: Updated flow with connections added between nodes.
    """
    for node, agent in flow.items():
        for other_node, other_agent in flow.items():
            if node == other_node:
                continue
            if agent["name"] in other_agent["context_vars"]:
                flow[node]["outputs"]["output"]["connections"].append({
                    "node": other_agent["id"],
                    "output": "input"
                })
                flow[other_node]["inputs"]["input"]["connections"].append({
                    "node": agent["id"],
                    "input": "output"
                })

    return flow

def get_positions(flow: dict, box_width : int=300, separation : int=500, initial_x : int=100, initial_y : int=100):

    # assign column number to each node
    columns = defaultdict(int)
    

def drawFlow(flow: Flow):
    data = {}
    drawflow = {"drawflow": {
        "Home": {
            "data": data}}}

    for agent in flow.__dict__["agents"]:
        data[agent.id] = {
            "id": agent.id,
            "name": agent.name,
            "context_vars": agent.context_vars,
            "class": "agent",
            "html": HTML.format(name=agent.name, task=agent.task),
            "typenode": False,
            "inputs": {"input": {"connections": []}},
            "outputs": {"output": {"connections": []}}

        }

    data = get_io(data)
    for graph in data:
        print(graph)
        print(find_longest_path_to_sources(data, graph.id))
        # get_positions(data)


flow = Flow().from_yaml("test_flow_yaml_v2.yaml", tools=[
    WikipediaTool(), DuckDuckGoSearchTool(), WebsiteTool()])

drawFlow(flow)
response = flow.run("Ecrit une courte biographie sur Emmanuel Macron")
print(response["summarizer"].content)
