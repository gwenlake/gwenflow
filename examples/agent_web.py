import dotenv

from gwenflow import Agent, ChatOpenAI, TelemetryBase, set_log_level_to_debug
from gwenflow.tools import PDFReaderTool, WebsiteReaderTool

set_log_level_to_debug()

dotenv.load_dotenv(override=True)

telemetry = TelemetryBase(service_name="gwenflow-v1", enabled=True, endpoint="http://localhost:6006/v1/traces")

telemetry.setup_telemetry()
telemetry.add_exporter()

agent = Agent(
    name="Website explorer",
    llm=ChatOpenAI(model="gpt-4o-mini"),
    tools=[WebsiteReaderTool(), PDFReaderTool()],
)

response = agent.run("Summarize the activity of the following company: https://gwenlake.com.")
print(response.content)
