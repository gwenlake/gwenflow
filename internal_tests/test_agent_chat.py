import context

import json
from gwenflow import Agent, ChatAgent, ChatOpenAI
from gwenflow.tools import WebsiteReaderTool, PDFTool, DuckDuckGoSearchTool, DuckDuckGoNewsTool, TavilyWebSearchTool
from gwenflow import set_log_level_to_debug

set_log_level_to_debug()

agent = Agent(
    name="News",
    llm=ChatOpenAI(model="gpt-4o-mini"),
    # reasoning_model=ChatOpenAI(model="o3-mini"),
    tools=[TavilyWebSearchTool(), WebsiteReaderTool(), PDFTool()],
    instructions = [
        "Always start with a Google Search.",
        "If you get a list of web links, systematically scrape the content of all the linked websites to extract detailed information about the topic.",
    ]
)

chat_agent = ChatAgent(
    agent=agent,
)

messages = [
    {
        "role": "user",
        "content": "hello",
        # "content": "Hello, ca va ?"
        # "content": "Create a list of known toxicities of targeting CD123, including linking to clinical trials or data where toxicity liabilities have been shown."
    }
]

response = chat_agent.run(messages)
print(response.content)
