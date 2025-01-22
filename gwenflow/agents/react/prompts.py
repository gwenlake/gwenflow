PROMPT_REACT = """
## Output Format

If you need to use a tool, please answer using the following format:
```
Thought: Your detailed reasoning about what to do next.
Action: tool name (one of {tool_names}) if using a tool.
Action Input: the input to the tool, in a JSON format representing the kwargs (e.g. {{"input": "hello world", "num_beams": 5}})
```

If you have enough information to answer the query:
```
Thought: Your final reasoning process
Final Answer: Your comprehensive answer to the query
```

If this format is used, the user will respond in the following format:
```
Observation: tool response
```

## Remember:
- Please use a valid JSON format for the Action Input. Do NOT do this {{'input': 'hello world', 'num_beams': 5}}.
- You should keep repeating the above format until you have enough information to answer the question without using any more tools.
"""
