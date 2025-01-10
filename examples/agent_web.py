import dotenv

from gwenflow import ChatOpenAI, Agent
from gwenflow.tools import WebsiteTool, PDFTool
from gwenflow.utils import set_log_level_to_debug


set_log_level_to_debug()

dotenv.load_dotenv(override=True)


agent = Agent(
    name="Website explorer",
    role="Analyse the content of a website",
    llm=ChatOpenAI(model="gpt-4o-mini"),
    tools=[WebsiteTool(), PDFTool()],
)

response = agent.run("Summarize the activity of the following company: https://gwenlake.com.")
print(response.content)
