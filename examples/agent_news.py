import dotenv

from gwenflow import ChatOpenAI, Agent
from gwenflow.tools import WebsiteTool, DuckDuckGoNewsTool
from gwenflow.utils import set_log_level_to_debug


set_log_level_to_debug()

dotenv.load_dotenv(override=True)


agent = Agent(
    name="News",
    role="You get some information about a topic.",
    llm=ChatOpenAI(model="gpt-4o-mini"),
    tools=[DuckDuckGoNewsTool(), WebsiteTool()],
    instructions = "If you get a list of web links, systematically scrape the content of all the linked websites to extract detailed information about the topic.",
)

response = agent.run("Tell me something about the recent plane crash in South Korea.")
print(response.content)
