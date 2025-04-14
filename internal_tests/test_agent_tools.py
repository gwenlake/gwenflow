import context
import requests
import json
import pandas as pd

from gwenflow import ChatOpenAI, Agent
from gwenflow.tools import WebsiteReaderTool, PDFReaderTool, DuckDuckGoSearchTool, DuckDuckGoNewsTool, TavilyWebSearchTool
from gwenflow import set_log_level_to_debug

set_log_level_to_debug()


# agent = Agent(
#     name="News",
#     llm=ChatOpenAI(model="gpt-4o-mini"),
#     # reasoning_model=ChatOpenAI(model="o3-mini"),
#     tools=[TavilyWebSearchTool(), WebsiteReaderTool(), PDFTool()],
#     instructions = [
#         "Always start with a Google Search.",
#         "If you get a list of web links, systematically scrape the content of all the linked websites to extract detailed information about the topic.",
#     ]
# )

# response = agent.run("Tell me something about the recent plane crash in South Korea.")
# response = agent.run("Retrieve the care pathway for AML in the USA and Europe. Where available, share link to the clinical guidelines. Answer in plain text (not in JSON).")
# response = agent.run("Create a list of known toxicities of targeting CD123, including linking to clinical trials or data where toxicity liabilities have been shown")
# print(response.content)

# agent = Agent(
#     name="News",
#     role="You get some information about a topic.",
#     llm=ChatOpenAI(model="gpt-4o-mini"),
#     tools=[DuckDuckGoNewsTool(), WebsiteReaderTool()],
#     instructions = "If you get a list of web links, systematically scrape the content of all the linked websites to extract detailed information about the topic.",
# )

# response = agent.run("Tell me something about the recent plane crash in South Korea.")
# print(response.content)
# exit(1)


agent = Agent(
    name="Assistant",
    instructions="Your are a helpful assistant.",
    tools=[WebsiteReaderTool(max_depth=2)],
    # tool_choice="required",
    # reasoning_model=ChatOpenAI(model="o3-mini"),
)

response = agent.run("Tell me something about Gwenlake using the following website: https://gwenlake.com")
print(response.content)

for source in response.sources:
    print("---")
    print(f"{source.name} ({source.id})")
    print(pd.DataFrame(source.data))
