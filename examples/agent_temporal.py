"""Run an agent durably on Temporal, with a tool that waits for a human approval.

pip install 'gwenflow[temporal]'
temporal server start-dev
python examples/agent_temporal.py

The model requests and the tool calls run as Temporal activities: kill the script while
the workflow waits for your approval, restart it, and the run resumes where it stopped
(same workflow id) without calling the model again.
"""

import asyncio
import json
from datetime import timedelta

import dotenv
from temporalio import workflow
from temporalio.client import Client, WorkflowExecutionStatus
from temporalio.common import RetryPolicy
from temporalio.exceptions import WorkflowAlreadyStartedError
from temporalio.worker import Worker

with workflow.unsafe.imports_passed_through():
    import requests

    from gwenflow import Agent, ChatOpenAI, Tool
    from gwenflow.durability import AgentWorkflow, TemporalDurability, worker_options

dotenv.load_dotenv(override=True)


def get_exchange_rate(currency_iso: str) -> str:
    """Get the current exchange rate against USD for a given currency. Currency MUST be in iso format."""
    response = requests.get("http://www.floatrates.com/daily/usd.json", timeout=10).json()
    data = response.get(currency_iso.lower())
    return json.dumps(data) if data else "Currency not found"


def book_fx_trade(currency_iso: str, amount_usd: float) -> str:
    """Book a trade converting an amount of USD into a currency.

    Args:
        currency_iso: ISO code of the currency to buy.
        amount_usd: Amount of USD to convert.
    """
    return f"Trade booked: {amount_usd} USD -> {currency_iso.upper()}"  # demo: nothing is booked


# A durable agent is an ordinary agent with a stable name (activity names derive from it)
# and a TemporalDurability. Outside a workflow, `agent.run()` behaves exactly as usual.
fx_agent = Agent(
    name="fx-desk",
    instructions=[
        "You are an FX desk assistant.",
        "Check the current rate before booking any trade, and give the rate you used.",
    ],
    llm=ChatOpenAI(model="gpt-4o-mini"),
    tools=[Tool(get_exchange_rate), Tool(book_fx_trade, requires_approval=True)],
    durability=TemporalDurability(
        # Per-tool activity options, merged over the defaults (60 s, Temporal's default retries).
        tool_activity_config={
            "get_exchange_rate": {
                "start_to_close_timeout": timedelta(seconds=20),
                "retry_policy": RetryPolicy(maximum_attempts=5),
            },
        },
    ),
)


@workflow.defn
class FxDeskWorkflow(AgentWorkflow):
    # Declared here, its activities are registered by worker_options().
    __gwenflow_agents__ = [fx_agent]

    @workflow.run
    async def run(self, prompt: str) -> str:
        # Pauses durably on each tool call that requires approval, until `approve` is signaled.
        response = await self.run_agent(fx_agent, prompt, timeout=timedelta(hours=1))
        return response.content


async def main():
    client = await Client.connect("localhost:7233")

    async with Worker(client, task_queue="fx-desk", **worker_options(FxDeskWorkflow)):
        try:
            handle = await client.start_workflow(
                FxDeskWorkflow.run,
                "Convert 1000 USD into euros.",
                id="fx-desk-demo",
                task_queue="fx-desk",
            )
        except WorkflowAlreadyStartedError:
            # Restarted while a run was in progress: pick it up where it stopped.
            handle = client.get_workflow_handle("fx-desk-demo")

        # The workflow exposes the tool calls it waits for; answer them with a signal.
        answered = set()
        while (await handle.describe()).status == WorkflowExecutionStatus.RUNNING:
            for request in await handle.query(FxDeskWorkflow.get_pending_approvals):
                if request["tool_call_id"] in answered:  # signaled, not processed yet
                    continue
                answered.add(request["tool_call_id"])
                answer = input(f"Run {request['tool_name']}({request['tool_args']})? [y/N] ").strip().lower()
                await handle.signal(FxDeskWorkflow.approve, args=[request["tool_call_id"], answer == "y"])
            await asyncio.sleep(0.5)

        print(await handle.result())


if __name__ == "__main__":
    asyncio.run(main())
