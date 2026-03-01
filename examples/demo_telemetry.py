import asyncio

from dotenv import load_dotenv

from gwenflow.agents.agent import Agent
from gwenflow.llms.openai.chat import ChatOpenAI
from gwenflow.telemetry.base import TelemetryBase
from gwenflow.tools.tavily import TavilyWebSearchTool

load_dotenv()

telemetry = TelemetryBase(service_name="gwenflow-dev",)
telemetry.initialize()

async def test_telemetry():

    llm = ChatOpenAI(
        model="gpt-5-nano",
    )

    agent = Agent(
        name="Football expert",
        llm=llm,
        tools=[TavilyWebSearchTool()]
    )
    async for i in agent.arun_stream("Whar was the last Rennes' match result ?"):
        print(i.content)


asyncio.run(test_telemetry())
