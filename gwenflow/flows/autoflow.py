
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


TASK_GENERATOR = """
You are an AI Agent creator tasked with generating a list of AI Agents based on a given set of tasks. \
    Your goal is to create a JSON array of Agent objects, each designed to accomplish a specific task within a larger objective.

Available Tools:
<tools>
{tools}
</tools>

Tasks to Accomplish:
<tasks>
{tasks}
</tasks>

Instructions for Creating AI Agents:

1. Analyze the given tasks and break them down into discrete steps that can be assigned to individual AI Agents.

2. For each task, create an Agent with the following properties:
   - name: A descriptive name for the Agent
   - role: A descriptive role for the Agent
   - task: A detailed description of what the Agent needs to accomplish
   - context: An array of Agent names whose results this Agent depends on (can be empty)
   - tools: An array of tools the Agent can use to complete its task (can be empty)
   - description: The task formulated as a question to ask the user

3. Adhere to these guidelines:
   - Create Agents based on the tasks provided
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

class AutoFlow(Flow):

    def run(self, user_prompt: str, output_file: Optional[str] = None) -> str:

        tools = [ tool.name for tool in self.tools ]
        tools = ", ".join(tools)

        task_prompt = TASK_GENERATOR.format(tasks=user_prompt, tools=tools, examples=json.dumps(EXAMPLE, indent=4))
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

        return super().run(user_prompt=user_prompt, output_file=output_file)
