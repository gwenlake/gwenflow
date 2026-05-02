import dotenv

from gwenflow import Agent, ChatOpenAI
from gwenflow.tools import PDFReaderTool, WebsiteReaderTool

dotenv.load_dotenv(override=True)

agent = Agent(
    name="Website explorer",
    llm=ChatOpenAI(model="gpt-4o-mini"),
    tools=[WebsiteReaderTool(), PDFReaderTool()],
)

response = agent.run("Summarize the activity of the following company: https://gwenlake.com.")
print(response.content)
