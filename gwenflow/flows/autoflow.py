
from typing import List, Callable, Union, Optional, Any, Dict
from pydantic import BaseModel
import json

from gwenflow.agents import Agent
from gwenflow.tasks import Task
from gwenflow.flows import Flow
from gwenflow.tools import Tool
from gwenflow.utils.json import parse_json_markdown
from gwenflow.utils import logger


EXAMPLE = [
    {
        "id": 1,
        "role": "Biographer",
        "task": "Write two paragraphs of max 500 words each",
        "tools": "wikipedia",
        "dependent_task_ids": [],
    },
    { 
        "id": 2,
        "role": "Summarizer", 
        "task": "List of 10 bullet points",
        "tools": None,
        "dependent_task_ids ":[1],
    },
    {
        "id": 3,
        "role": "Generate a list of related topics",
        "task": "Generate a list of 5 related topics",
        "tools": None,
        "dependent_task_ids ":[2],
    },
    {
        "id":4,
        "role": "Final Report",
        "task": "Produce a final report in a Powerpoint file (pptx format)",
        "tools": "python",
        "dependent_task_ids ":[1,2,3],
    }
]


TASK_GENERATOR = """
You are an expert in creating a list of AI agents as a JSON array.

# Guidelines:
- Create new agents based on the objective.
- Limit agents to those that can be completed with the available tools listed below.
- Role should be give.
- Tasks should be detailed.
- Current tool options are {tools}.
- When requiring multiple searches, use the tools multiple times. This tool will use the dependent task result to generate the search query if necessary.
- Use [user-input] sparingly and only if you need to ask a question to the user who set up the objective.
- The task description should be the question you want to ask the user.
- dependent_task_ids should always be an empty array, or an array of numbers representing the task ID it should pull results from.
- Make sure all task IDs are in chronological order.
- You can use multiple tools for a single task by separating them with a comma.

# Example:
Objective: Look up AI news from today (May 27, 2023) and prepare a report.
Task list
```json
{examples}
```

# Your task:
Objective: {objective}
Agent list:
"""


class AutoFlow(Flow):

    manager: List[Agent] = []
    llm: Any = None
    tools: List[Tool] = []

    def execute_task(self, user_prompt: str):

        tools = [ tool.name for tool in self.tools ]
        tools = ", ".join(tools)

        task_prompt = TASK_GENERATOR.format(objective=user_prompt, tools=tools, examples=json.dumps(EXAMPLE, indent=4))

        agents_json = self.llm.invoke(messages=[{"role": "user", "content": task_prompt}])
        agents_json = parse_json_markdown(agents_json)

        for agent_json in agents_json:

            tools = []

            if agent_json.get("tools"):
                task_tools = agent_json["tools"].split(",")
                for t in self.tools:
                    if t.name in task_tools:
                        tools.append(t)

            agent = Agent(
                llm=self.llm,
                role=agent_json.get("role"),
                task=agent_json.get("task"),
                tools=tools,
            )

            self.agents.append(agent)

        return self.run(user_prompt)
