import context
import requests
import json

from gwenflow import ChatOpenAI, Agent, Retriever
from gwenflow.readers import WebsiteReader
from gwenflow.tools import RetrieverTool, WebsiteReaderTool, PDFReaderTool, DuckDuckGoSearchTool, DuckDuckGoNewsTool, TavilyWebSearchTool
from gwenflow import set_log_level_to_debug

set_log_level_to_debug()

# retriever = Retriever(name="knowledge-test-gwenlake")
# reader = WebsiteReader(delay=False)
# documents = reader.read("https://gwenlake.com")
# retriever.load_documents(documents)

agent = Agent(
    name="base-agent",
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.7),
    # retriever=Retriever(name="knowledge-test-gwenlake"),
    tools=[RetrieverTool(retriever=Retriever(name="knowledge-test-gwenlake", top_k=5))]
)

# response = agent.run("Tell me something about Gwenlake using the following website: https://gwenlake.com")
# print(response.content)

response = agent.run_stream("Tell me about Gwenlake.")
for chunk in response:
    if chunk.thinking:
        print(chunk.thinking)
    if chunk.content:
        print(chunk.content, end="")
