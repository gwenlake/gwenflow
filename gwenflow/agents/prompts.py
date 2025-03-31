PROMPT_KNOWLEDGE = """\
## Use the following references from the knowledge base if it helps:

<references>
{references}
<references>
"""

PROMPT_CONTEXT = """\
## Use the following context if it helps:

<context>
{context}
</context>
"""

PROMPT_STEPS = """
## Before providing your final answer, follow these steps:

<steps>
{reasoning_steps}
</steps>

This structured approach will help you organize your thoughts and ensure a thorough response.
"""

PROMPT_PREVIOUS_INTERACTIONS = """\
## Answer the question considering the previous interactions:

<previous_interactions>
{previous_interactions}
</previous_interactions>
"""

PROMPT_JSON_SCHEMA = """\
## Provide your output using the following JSON schema:

<json_schema>
{json_schema}
</json_schema>
"""

PROMPT_REASONING_STEPS_TOOLS = """\
Your objective is to thoroughly research your task using the following tools as your primary source and provide a detailed and informative answer.

You have access to the following tools:

<tools>
{tools}
</tools>

Please provide the detailed steps that are needed to achieve your task accurately and efficiently.
"""

PROMPT_REASONNING = """\
I am building an Agent. You are the thinker of the agent.  
Instructions:
- Provide a structured approach to solve the task.
- Enumerate your steps in one list.
- DO NOT answer to task.
- You are answering to a large language model.

Task: {task}
Answer:"""
