import asyncio
import json

from gwenflow.agents import Agent
from gwenflow.flows.flows import Flow, execute_flow
from gwenflow.flows.handlers import compose_task
from gwenflow.logger import logger
from gwenflow.registry import model_id, tool_key
from gwenflow.utils.json import parse_json_markdown

EXAMPLE = [
    {
        "name": "Biographer",
        "task": "Write two paragraphs of max 500 words each",
        "tools": ["wikipedia"],
        "depends_on": [],
    },
    {
        "name": "Summarizer",
        "task": "List of 10 bullet points",
        "tools": None,
        "depends_on": ["Biographer"],
    },
    {
        "name": "RelatedTopics",
        "task": "Generate a list of 5 related topics",
        "tools": None,
        "depends_on": ["Summarizer"],
    },
    {
        "name": "Final Report",
        "task": "Produce a final report in a Powerpoint file (pptx format)",
        "tools": ["wikipedia", "python"],
        "depends_on": ["Biographer", "Summarizer", "RelatedTopics"],
    },
]


TASK_GENERATOR = """
You are an AI Agent creator tasked with generating a list of AI Agents based on a given set of tasks.
Your goal is to create a JSON array of Agent objects, each designed to accomplish a specific task within a larger objective.

Available Tools:
<tools>
{tools}
</tools>

Tasks to Accomplish:
<tasks>
{tasks}
</tasks>

Instructions for creating AI Agents:

1. Analyze the given tasks and break them down into discrete steps that can be assigned to individual AI Agents.

2. For each task, create an Agent with the following properties:
   - name: A descriptive name for the Agent
   - task: A detailed description of what the Agent needs to accomplish
   - depends_on: An array of Agent names whose results this Agent depends on (can be empty)
   - tools: An array of tools the Agent can use to complete its task (can be empty)

3. Adhere to these guidelines:
   - Create Agents based on the tasks provided
   - Do not call any tool provided in this prompt. Just create the list of AI Agents
   - Limit Agents to those that can be completed with the available tools listed above
   - Ensure all tasks are detailed and specific
   - When multiple searches are required, use the tools multiple times
   - Do not use the same tool with the same arguments multiple times
   - Make sure all tasks are in chronological order
   - Validate the output of other Agents and reassign tasks if necessary

4. Wrap the following process in <agent_creation_process> tags for each Agent:
   a. List out all tasks and number them for organization
   b. Identify the specific task the Agent needs to accomplish
   c. Determine a suitable name for the Agent
   d. Write a detailed task description
   e. Consider and list potential dependencies on other tasks/Agents
   f. Identify any dependencies on other Agents (context)
   g. Select appropriate tools from the available options
   h. Formulate the task description as a question for the user

5. After creating all Agents, review the entire list to ensure:
   - All tasks are covered
   - Agents are in the correct chronological order
   - Dependencies (context) are correctly specified
   - Tool usage is appropriate and necessary

6. Generate the final JSON array of Agent objects.

Output Format:
The output should be a JSON array of Agent objects. Here's a generic example of the structure:

```json
{examples}
```

Remember to replace the generic content with actual Agent data based on the given tasks.

Now, please create the list of AI Agents based on the provided tasks.
"""


class AutoFlow:
    """Ask an LLM to plan a team of agents for a query, then run it as a `Flow`.

    `plan()` returns the flow without running it: it can be saved as YAML,
    reviewed, edited, or run durably like any other flow (built-in llm and tools
    are referenced by name; others need `run()`, which uses the instances).
    """

    def __init__(self, llm, tools=None):
        self.llm = llm
        self.tools = tools or []

    def plan(self, query: str) -> Flow:
        tool_names = ", ".join(t.name for t in self.tools)
        task_prompt = TASK_GENERATOR.format(
            tasks=query,
            tools=tool_names,
            examples=json.dumps(EXAMPLE, indent=4),
        )
        response = self.llm.invoke(input=[{"role": "user", "content": task_prompt}])
        agent_plan = parse_json_markdown(response.content)

        names = {entry["name"] for entry in agent_plan}
        nodes, connections = [], {}
        for entry in agent_plan:
            wanted = entry.get("tools") or []
            nodes.append(
                {
                    "name": entry["name"],
                    "type": "gwenflow.Agent",
                    "parameters": {
                        "name": entry["name"],
                        **self._model_parameter(),
                        "task": entry.get("task", ""),
                        "tools": [self._tool_reference(t) for t in self.tools if t.name in wanted],
                    },
                }
            )
            for dep in entry.get("depends_on") or []:
                if dep in names:
                    connections.setdefault(dep, [[]])[0].append({"node": entry["name"]})
        logger.info(f"AutoFlow planned {len(nodes)} agent(s).")
        return Flow.from_dict({"name": f"AutoFlow: {query[:60]}", "nodes": nodes, "connections": connections})

    def run(self, query: str) -> str:
        """Plan, then run the agents and return the last one's output.

        Agents run in-process with this AutoFlow's own llm and tool instances
        (custom ones included); independent agents run concurrently.
        """
        flow = self.plan(query)
        tools = {self._tool_reference(t): t for t in self.tools}

        async def run_agent(node, input_data):
            agent = Agent(
                name=node.parameters["name"],
                llm=self.llm,
                tools=[tools[ref] for ref in node.parameters.get("tools", []) if ref in tools],
            )
            task = compose_task(node.parameters.get("task", ""), input_data)
            logger.info(f"Running agent: {agent.name}")
            response = await asyncio.to_thread(agent.run, task)
            return response.content

        results = asyncio.run(execute_flow(flow, run_agent))
        return results[flow.topological_order()[-1]]

    def _model_parameter(self) -> dict:
        """`model` for the flow file, when the llm is a built-in provider."""
        try:
            return {"model": model_id(self.llm)}
        except ValueError:
            return {}

    @staticmethod
    def _tool_reference(tool) -> str:
        """Built-in tools by their registry key (portable to YAML), others by name."""
        try:
            return tool_key(tool)
        except ValueError:
            return tool.name
