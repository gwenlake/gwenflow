import context
import requests
import json
import pandas as pd

from gwenflow import ChatOpenAI, Agent, Message
from gwenflow.tools import WikipediaTool, TavilyWebSearchTool, WebsiteReaderTool, PDFReaderTool
from gwenflow import set_log_level_to_debug


set_log_level_to_debug()

chat_history = [
    Message(role="user", content="hello, my name is sylvain"),
    Message(role="assistant", content="hello. How can I help you?"),
    Message(role="user", content="Yes, I am interesting in news about France."),
]

agent = Agent(
    name="Biographer",
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.7),
    tools=[TavilyWebSearchTool(max_results=20)],
    # tools=[PDFReaderTool()],
    # tools=[WebsiteReaderTool(max_depth=2)],
    # tools=[TavilyWebSearchTool(), WikipediaTool(), WebsiteReaderTool()],
    # reasoning_model=ChatOpenAI(model="o3-mini"),
    # tool_choice="required"
)

# response = agent.run_stream("Tell me something about Emmanuel Macron and give me some recent news about him. When done, investigate more on one specific recent news. And please, talk to me using my name.")
# response = agent.run_stream("Tell me something about Emmanuel Macron.")
response = agent.run_stream("Tell me something about Trump in April 2025")
# response = agent.run_stream("Tell me something about Gwenlake using the following website https://gwenlake.com")
# response = agent.run_stream("Summarize this report https://www.ipcc.ch/report/ar6/wg2/downloads/report/IPCC_AR6_WGII_SummaryForPolicymakers.pdf")
# response = agent.run_stream("Summarize this report ./documents/2021-tesla-impact-report.pdf")

for chunk in response:
    # print(chunk)
    if chunk.thinking:
        print(chunk.thinking)
    if chunk.content:
        print(chunk.content, end="")

print("")

for source in chunk.sources:
    print("---")
    print(f"{source.name} ({source.id})")
    print(pd.DataFrame(source.data))

# for m in chunk.output:
#     if m.role == "tool":
#         print(pd.DataFrame(json.loads(m.content)))

# response = agent.run("Hello", chat_history=chat_history, stream=True)
# for chunk in response:
#     # print(chunk)
#     if chunk.thinking:
#         print(chunk.thinking)
#     if chunk.delta:
#         print(chunk.delta, end="")
