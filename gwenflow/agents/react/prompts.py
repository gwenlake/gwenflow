PROMPT_REACT = """
## Output Format

If you need to use a tool, please answer using the following format:

```
Thought: you should always think about what to do.
Action: tool name (one of {tool_names}) if using a tool.
Action Input: the input to the tool, in a JSON format representing the kwargs (e.g. {{"input": "hello world", "num_beams": 5}})
```

Please ALWAYS start with a Thought.

Please use a valid JSON format for the Action Input. Do NOT do this {{'input': 'hello world', 'num_beams': 5}}.

If this format is used, the user will respond in the following format:

```
Observation: tool response
```

You should keep repeating the above format until you have enough information to answer the question without using any more tools. At that point, you MUST respond in the following two format:

```
Thought: the final thought to the original input question
Final Answer: the final answer to the original input question
```

## Remember:
- Please use a valid JSON format for the Action Input. Do NOT do this {{'input': 'hello world', 'num_beams': 5}}.
- You should keep repeating the above format until you have enough information to answer the question without using any more tools.
"""
