
from typing import List, Callable, Union, Optional, Any, Dict
from pydantic import BaseModel
import json

from gwenflow.agents import Agent
from gwenflow.flows import Flow
from gwenflow.tools import BaseTool
from gwenflow.utils.json import parse_json_markdown
from gwenflow.utils import logger


EXAMPLE = [
    {
        "name": "Biographer",
        "role": "Write two paragraphs of max 500 words each",
        "tools": ["wikipedia"],
        "context": [],
    },
    { 
        "name": "Summarizer",
        "role": "List of 10 bullet points",
        "tools": None,
        "context": ["Biographer"],
    },
    {
        "name": "RelatedTopics",
        "role": "Generate a list of 5 related topics",
        "tools": None,
        "context": ["Summarizer"],
    },
    {
        "name": "Final Report",
        "role": "Produce a final report in a Powerpoint file (pptx format)",
        "tools": ["wikipedia","python"],
        "context": ["Biographer","Summarizer","RelatedTopics"],
    }
]


TASK_GENERATOR = """You are an expert in creating a list of AI agents as a JSON array.

# Guidelines:
- Create agents based on the objective.
- A name should be given to Agents.
- Limit agents to those that can be completed with the available tools listed below.
- Tasks should be detailed.
- When requiring multiple searches, use the tools multiple times. This tool will use the dependent task result to generate the search query if necessary.
- The task description should be the question you want to ask the user.
- Make sure all task are in chronological order.
- [context] should always be an empty array or an array of Agent names it should pull results from.
- Current tool options are {tools}.
- [tools] should always be an empty array or an array of Tools the Agent can use to complete its task.

# Example:
Objective: Look up AI news from today (May 27, 2023) and prepare a report.
Agent list
```json
{examples}
```

# Your task:
Objective: {objective}
Agent list:
"""


class AutoFlow(Flow):

    llm: Any = None
    tools: List[BaseTool] = []

    def run(self, query: str) -> str:

        tools = [ tool.name for tool in self.tools ]
        tools = ", ".join(tools)

        task_prompt = TASK_GENERATOR.format(objective=query, tools=tools, examples=json.dumps(EXAMPLE, indent=4))
        
        agents_json = self.llm.invoke(messages=[{"role": "user", "content": task_prompt}])
        agents_json = parse_json_markdown(agents_json)

        for agent_json in agents_json:

            tools = []
            if agent_json.get("tools"):
                for t in self.tools:
                    if t.name in agent_json["tools"]:
                        tools.append(t)

            agent = Agent(
                llm=self.llm,
                name=agent_json.get("name"),
                role=agent_json.get("role"),
                tools=tools,
                context_vars=agent_json.get("context"),
            )

            self.agents.append(agent)

        self.describe()

        return super().run(query)
