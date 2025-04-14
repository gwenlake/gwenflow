import context
import json

from gwenflow import ChatOpenAI, Agent, FunctionTool
from gwenflow import set_log_level_to_debug

set_log_level_to_debug()

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
wikipedia   = WikipediaQueryRun(api_wrapper=api_wrapper)

tool_wikipedia = FunctionTool.from_langchain( wikipedia )

agent = Agent(
    name="Helpful Analyst",
    instructions=["Get some useful information about my request", "Answer as precisely as possible."],
    llm=ChatOpenAI(model="gpt-4o-mini"),
    tools=[tool_wikipedia],
)

response = agent.run("Summarize the wikipedia's page about Winston Churchill.")
print(response.content)
exit(1)

# print("\n----------------------------")
# query = "Give me a list of 5 documents about Winston Churchill, in JSON format. Please include a title, an url and a quick summary."
# task = Task(
#     description=query,
#     expected_output="Answer as precisely as possible.",
#     agent=agent
# )

# print("Q:", query)
# print("A:", task.run())


# print("\n----------------------------")
# query = "What is the weather in Paris?"
# task = Task(
#     description=query,
#     expected_output="Answer as precisely as possible.",
#     agent=agent
# )

# print("Q:", query)
# print("A:", task.run())
