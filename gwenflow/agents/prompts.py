PROMPT_TOOLS = """
## Tools
Your objective is to thoroughly research your task using the following tools as your primary source and provide a detailed, informative answer.

You have access to the following tools:
<tools>
{tools}
</tools>

The tool call you write is an 'Action'. After the tool is executed, you will get the result of the tool call as an 'Observation'.
This Action/Observation can repeat N times. You should take several steps when needed.
You can use the result of the previous 'Action' as input for the next 'Action'.
The 'Observation' will always be a string. Then you can use it as input for the next 'Action'.
"""

PROMPT_STEPS = """Before providing your final answer, follow these steps:

1. List the key topics or concepts you've identified from the task.
2. For each topic, list potential information you plan to search.
3. For each information:
   a. Note the most relevant information you find.
   b. Include any important quotes, with proper citation.
   c. Highlight any statistical data or key facts.
4. Identify any contradictions or gaps in the information you've found.
5. Summarize your key findings and how they relate to the task.

This structured approach will help you organize your thoughts and ensure a thorough response.
"""


PROMPT_TASK_REACT = """
## Output Format
To answer the question, please use the following format.

```
Thought: I need to use a tool to help me answer the question.
Action: tool name (one of {tool_names}) if using a tool.
Action Input: the input to the tool, in a JSON format representing the kwargs (e.g. {{"input": "hello world", "num_beams": 5}})
```

Please ALWAYS start with a Thought.

Please use a valid JSON format for the Action Input. Do NOT do this {{'input': 'hello world', 'num_beams': 5}}.

If this format is used, the user will respond in the following format:

```
Observation: tool response
```

You should keep repeating the above format until you have enough information
to answer the question without using any more tools. At that point, you MUST respond
in the one of the following two formats:

```
Thought: I can answer accurately without using any more tools.
Final Answer: [your answer here]
```

```
Thought: I cannot answer the question with the provided tools.
Final Answer: Sorry, I cannot answer your query.
```

## Current Conversation
Below is the current conversation consisting of interleaving human and assistant messages.

{task}
"""
