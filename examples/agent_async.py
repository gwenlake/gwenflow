import asyncio

import dotenv

from gwenflow import Agent, ChatOpenAI
from gwenflow.tools import DuckDuckGoSearchTool, WikipediaTool

dotenv.load_dotenv(override=True)


async def ask(agent: Agent, question: str) -> None:
    print(f"\nQ: {question}")
    print("A: ", end="", flush=True)
    async for chunk in agent.arun_stream(question):
        if chunk.content:
            print(chunk.content, end="", flush=True)
    print()


async def main():
    agent = Agent(
        name="Research Assistant",
        instructions="Answer questions concisely using your tools when needed.",
        llm=ChatOpenAI(model="gpt-4o-mini"),
        tools=[DuckDuckGoSearchTool(), WikipediaTool()],
    )

    questions = [
        "What is the capital of Australia?",
        "Who invented the World Wide Web?",
        "What is the latest version of Python?",
    ]

    await asyncio.gather(*[ask(agent, q) for q in questions])


asyncio.run(main())
