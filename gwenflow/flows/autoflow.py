
from typing import List, Callable, Union, Optional, Any, Dict
from pydantic import BaseModel
import logging
import json

from gwenflow.tools import Tool
from gwenflow.tasks import Task
from gwenflow.flows import Flow



logger = logging.getLogger(__name__)


EXAMPLE = [
    {
        "id": 1,
        "task": "AI news today",
        "tool": "web-search ",
        "dependent_task_ids":[],
        "status": "incomplete ",
        "result": None,
        "result_summary":None
    },
    { 
        "id":2,
        "task": "Extract key points from AI news articles ", 
        "tool": "text-completion ", 
        "dependent_task_ids ":[1],
        "status": "incomplete ",
        "result":None,
        "result_summary":None
    },
    {
        "id":3,
        "task": "Generate a list of AI-related words and phrases ",
        "tool": "text-completion ",
        "dependent_task_ids ":[2],
        "status": "incomplete ",
        "result":None,
        "result_summary":None
    },
    {
        "id":4,
        "task": "Write a poem using AI-related words and phrases ",
        "tool": "text-completion ",
        "dependent_task_ids ":[3],
        "status": "incomplete ",
        "result":None,
        "result_summary":None
    },
    {
        "id":5,
        "task": "Final summary report",
        "tool": "text-completion",
        "dependent_task_ids ":[1,2,3,4],
        "status": "incomplete",
        "result": None,
        "result_summary": None
    }
]


TASK_GENERATOR = """\
You are a task manager AI. You are an expert task creation AI tasked with creating a list of tasks as a JSON array.
Create new tasks based on the objective.
Limit tasks types to those that can be completed with the available tools listed below. Task description should be detailed.
Current tool options are {tools}.
Result will be a summary of relevant information from the first few articles.
When requiring multiple searches, use the tools multiple times. This tool will use the dependent task result to generate the search query if necessary.
Use [user-input] sparingly and only if you need to ask a question to the user who set up the objective.
The task description should be the question you want to ask the user.
dependent_task_ids should always be an empty array, or an array of numbers representing the task ID it should pull results from.
Make sure all task IDs are in chronological order.

# Example
Objective: Look up AI news from today (May 27, 2023) and write a poem.
Task list
```json
{examples}
```

# Your task
Objective: {objective}
Task list:
"""


class AutoFlow(Flow):

    # @model_validator(mode="before")
    # @classmethod
    # def get_input_variables(cls, values: dict) -> Any:
    #     """Get input variables."""
    #     all_variables = set()
    #     for prompt in values["prompts"]:
    #         all_variables.update(prompt.input_variables)
    #     values["input_variables"] = list(all_variables)
    #     return values
    
    def generate_tasks(self, llm: Any, objective: str, tools: str):
        llm = llm
        task_prompt = TASK_GENERATOR.format(objective=objective, tools=tools, examples=json.dumps(EXAMPLE, indent=4))
        tasks_json = llm.invoke(messages=[{"role": "user", "content": task_prompt}])
        return tasks_json

