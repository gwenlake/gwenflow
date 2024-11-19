
from typing import List, Callable, Union, Optional, Any, Dict
from pydantic import BaseModel
import logging
import json

# from gwenflow.tools import Tool
from gwenflow.agents import Agent
from gwenflow.tasks import Task
from gwenflow.flows import Flow
from gwenflow.utils.json import parse_json_markdown


logger = logging.getLogger(__name__)


EXAMPLE = [
    {
        "id": 1,
        "description": "A biography of Marilyn Monroe",
        "expected_output": "Two paragraphs of max 500 words each.",
        "tools": "wikipedia",
        "dependent_task_ids": [],
    },
    { 
        "id": 2,
        "description": "Summarize the key elements in the biography.", 
        "expected_output": "List of 10 bullet points.",
        "tools": None,
        "dependent_task_ids ":[1],
    },
    {
        "id": 3,
        "description": "Generate a list of related topics.",
        "expected_output": "List 5 bullet points.",
        "tools": None,
        "dependent_task_ids ":[2],
    },
    {
        "id":4,
        "description": "Write a poem these informations",
        "expected_output": "A poems of two pages.",
        "tools": None,
        "dependent_task_ids ":[3],
    },
    {
        "id":5,
        "description": "Final summary report",
        "expected_output": "A Powerpoint presentation in pptx format.",
        "tools": "python",
        "dependent_task_ids ":[1,2,3,4],
    }
]


TASK_GENERATOR = """
You are an expert task creation AI tasked with creating a list of tasks as a JSON array.

RULES:
----
- Create new tasks based on the objective.
- Limit tasks types to those that can be completed with the available tools listed below.
- Task description should be detailed.
- Current tool options are {tools}.
- When requiring multiple searches, use the tools multiple times. This tool will use the dependent task result to generate the search query if necessary.
- Use [user-input] sparingly and only if you need to ask a question to the user who set up the objective.
- The task description should be the question you want to ask the user.
- dependent_task_ids should always be an empty array, or an array of numbers representing the task ID it should pull results from.
- Make sure all task IDs are in chronological order.
- You can use multiple tools for a single task by separating them with a comma.

EXAMPLE:
----
Objective: Look up AI news from today (May 27, 2023) and write a poem.
Task list
```json
{examples}
```

YOUR JOB:
----
Objective: {objective}
Task list:
"""


class AutoFlow(Flow):

    
    def generate_tasks(self, objective: str):

        tools = [ tool.name for tool in self.tools ]
        tools = ",".join(tools)

        task_prompt = TASK_GENERATOR.format(objective=objective, tools=tools, examples=json.dumps(EXAMPLE, indent=4))
        tasks = self.llm.invoke(messages=[{"role": "user", "content": task_prompt}])
        tasks = parse_json_markdown(tasks)

        for task in tasks:

            _tools = []

            if task.get("tools"):
                task_tools = task["tools"].split(",")
                for tool in self.tools:
                    if tool.name in task_tools:
                        _tools.append(tool)

            _agent = Agent(
                role="Generic Agent.",
                instructions="You are a helpful AI agent..",
                llm=self.llm,
                tools=_tools,
            )

            _task = Task(
                description=task["description"],
                expected_output=task["expected_output"],
                agent=_agent
            )
            self.tasks.append(_task)

        return self.tasks
