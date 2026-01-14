import dotenv

from gwenflow import Agent, ChatOpenAI, set_log_level_to_debug
from gwenflow.tools import DuckDuckGoNewsTool, WebsiteReaderTool

set_log_level_to_debug()

dotenv.load_dotenv(override=True)


agent = Agent(
    name="News",
    llm=ChatOpenAI(model="gpt-4o-mini"),
    tools=[DuckDuckGoNewsTool(), WebsiteReaderTool()],
    instructions="If you get a list of web links, systematically scrape the content of all the linked websites to extract detailed information about the topic.",
)

response = agent.run("Tell me something about the recent plane crash in South Korea.")
print(response.content)
