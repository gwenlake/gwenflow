import json

import networkx as nx

from gwenflow.agents import Agent
from gwenflow.logger import logger
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
    def __init__(self, llm, tools=None):
        self.llm = llm
        self.tools = tools or []

    def run(self, query: str) -> str:
        tool_names = ", ".join(t.name for t in self.tools)

        task_prompt = TASK_GENERATOR.format(
            tasks=query,
            tools=tool_names,
            examples=json.dumps(EXAMPLE, indent=4),
        )
        response = self.llm.invoke(input=[{"role": "user", "content": task_prompt}])
        agent_plan = parse_json_markdown(response.content)

        agents: dict[str, dict] = {}
        for entry in agent_plan:
            agent_tools = []
            if entry.get("tools"):
                for t in self.tools:
                    if t.name in entry["tools"]:
                        agent_tools.append(t)
            agents[entry["name"]] = {
                "agent": Agent(name=entry["name"], llm=self.llm, tools=agent_tools),
                "task": entry.get("task", ""),
                "depends_on": entry.get("depends_on") or [],
            }

        graph = nx.DiGraph()
        for name in agents:
            graph.add_node(name)
        for name, info in agents.items():
            for dep in info["depends_on"]:
                if dep in agents:
                    graph.add_edge(dep, name)

        results: dict[str, str] = {}
        last_result = ""
        for name in nx.topological_sort(graph):
            info = agents[name]
            task = info["task"]
            context_parts = [
                f"Output from {dep}:\n{results[dep]}"
                for dep in info["depends_on"]
                if dep in results
            ]
            if context_parts:
                task = f"{task}\n\nContext from previous steps:\n" + "\n\n".join(context_parts)
            logger.info(f"Running agent: {name}")
            response = info["agent"].run(task)
            results[name] = response.content
            last_result = response.content

        return last_result
