import context

import json
from gwenflow import Agent
from gwenflow import set_log_level_to_debug

set_log_level_to_debug()

instructions = [
    "Your goal is to reason about the task or query and decide on the best course of action to answer it accurately.",
    "You are a meticulous and thoughtful assistant that solves a problem by thinking through it step-by-step.",
    "If you cannot find the necessary information after using available tools, admit that you don't have enough information to answer the query confidently.",
]

agent = Agent(
    name="agent-wikipedia",
    instructions="You help me with general questions."
)

response = agent.run("Tell me something about Riemann. Search on wikipedia please.")
# # response = agent.run("Tell me something about Riemann", output_file="test.md")
print(response.content)
exit(1)

# print(agent.memory.get())

response = agent.run("Write three paragraphs about Winston Churchill.", stream=True)
for chunk in response:
    if chunk.delta:
        print(chunk.delta, end="")

exit(1)

# response = agent.run("One paragraph about Winston Churchill?")
# print("================== INFOS 1")
# print(response.content)

# response = agent.run("Another paragraph about this guy")
# print("================== INFOS 2")
# print(response.content)

# response = agent.run("3 other ones?", stream=True)
# print("================== INFOS 3")
# for chunk in response:
#     if isinstance(chunk, str):
#         print(chunk, end="")

# print(agent.memory.get())
