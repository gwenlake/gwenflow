import context
import requests
import json

from gwenflow import ChatOpenAI, Agent, Task, Tool
from gwenflow.utils import set_log_level_to_debug

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper


set_log_level_to_debug()

# A wrapper for wikipedia
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
tool_wikipedia = Tool.from_langchain( WikipediaQueryRun(api_wrapper=api_wrapper) )




llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

agent = Agent(
    llm=llm,
    role="Agent",
    instructions="You write biographies.",
    tools=[tool_wikipedia],
)

response = agent.run("Write a paragraph about Winston Churchill.")
print(response.content)
# print(json.dumps(response.messages, indent=4))

# response = agent.stream("Write two paragraphs about Winston Churchill.")
# for chunk in response:
#     if isinstance(chunk, str):
#         print(chunk, end="")
