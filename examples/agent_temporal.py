"""Run an agent durably on Temporal: the simplest setup.

pip install 'gwenflow[temporal]'
temporal server start-dev
python examples/agent_temporal.py
"""

import asyncio
import json

import dotenv
from temporalio import workflow
from temporalio.client import Client
from temporalio.worker import Worker

with workflow.unsafe.imports_passed_through():
    import requests

    from gwenflow import Agent, ChatOpenAI, Tool
    from gwenflow.durability import TemporalDurability, worker_options

dotenv.load_dotenv(override=True)


def get_exchange_rate(currency_iso: str) -> str:
    """Get the current exchange rate against USD for a given currency. Currency MUST be in iso format."""
    response = requests.get("http://www.floatrates.com/daily/usd.json", timeout=10).json()
    data = response.get(currency_iso.lower())
    return json.dumps(data) if data else "Currency not found"


# An ordinary agent: durable only when it runs inside a workflow.
agent = Agent(
    name="fx",
    instructions="Answer in one sentence and mention the date of the rate.",
    llm=ChatOpenAI(model="gpt-4o-mini"),
    tools=[Tool(get_exchange_rate)],
    durability=TemporalDurability(),
)


@workflow.defn
class ExchangeRates:
    @workflow.run
    async def run(self, currency: str) -> str:
        await agent.arun(f"What's the exchange rate of {currency}?")  # model and tool calls are activities
        response = await agent.arun("And the yen?")  # same run, same history
        return response.content


async def main():
    client = await Client.connect("localhost:7233")

    async with Worker(client, task_queue="agents", **worker_options(ExchangeRates, agents=[agent])):
        print(await client.execute_workflow(ExchangeRates.run, "EUR", id="exchange-rates", task_queue="agents"))


if __name__ == "__main__":
    asyncio.run(main())
