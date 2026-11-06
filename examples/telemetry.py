import asyncio

import dotenv

from gwenflow import Agent, ChatOpenAI, Telemetry
from gwenflow.telemetry import tracer
from gwenflow.tools import TavilyWebSearchTool

dotenv.load_dotenv(override=True)
Telemetry()


async def main():
    llm = ChatOpenAI(model="gpt-4o-mini")
    agent = Agent(name="Football expert", llm=llm, tools=[TavilyWebSearchTool()])

    with tracer.session(session_id="Match result"):
        async for chunk in agent.arun_stream("What was the last Rennes match result?"):
            if chunk.content:
                print(chunk.content, end="", flush=True)
    print()


asyncio.run(main())
