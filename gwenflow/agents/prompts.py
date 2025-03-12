PROMPT_GUIDELINES = """
## Guidelines:
{guidelines}
"""

PROMPT_STEPS = """
## Before providing your final answer, follow these steps:

<steps>
{reasoning_steps}
</steps>

This structured approach will help you organize your thoughts and ensure a thorough response.
"""


PROMPT_TASK = """
## You have received the following task from your manager:

<task>
{task}
</task>
"""

PROMPT_TOOLS = """
## Tools
Your objective is to thoroughly research your <task> using the following tools as your primary source and provide a detailed and informative answer.

You have access to the following tools:

<tools>
{tools}
</tools>
"""

PROMPT_JSON_SCHEMA = """
## Provide your output using the following JSON schema:

<json_schema>
{json_schema}
</json_schema>
"""

PROMPT_REASONING_STEPS_TOOLS = """
Your objective is to thoroughly research your task using the following tools as your primary source and provide a detailed and informative answer.

You have access to the following tools:

<tools>
{tools}
</tools>

Please provide the detailed steps that are needed to achieve your task accurately and efficiently.
"""

PROMPT_TOOLS_REACT_GUIDELINES = """
The tool call you write is an 'Action'. After the tool is executed, you will get the result of the tool call as an 'Observation'.
This Action/Observation can repeat N times. You should take several steps when needed.
You can use the result of the previous 'Action' as input for the next 'Action'.
The 'Observation' will always be a string. Then you can use it as input for the next 'Action'.
"""

PROMPT_REASONNING = """
I am building a reAct agent. You are the thinker of the agent.  
Instructions:
- Given <task>, provide a very simple structured approach to solve the <task>.
- Enumerate your steps in one list.
- DO NOT answer to <task>.
- You are answering to a large language model.
"""