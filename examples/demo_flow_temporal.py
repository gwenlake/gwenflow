"""Run a YAML flow durably on Temporal, then schedule it every morning.

pip install 'gwenflow[temporal]'
temporal server start-dev
python examples/demo_flow_temporal.py
"""

import asyncio
from datetime import timedelta

import dotenv
from temporalio import workflow
from temporalio.client import Client, Schedule, ScheduleActionStartWorkflow, ScheduleIntervalSpec, ScheduleSpec
from temporalio.worker import Worker

with workflow.unsafe.imports_passed_through():
    from gwenflow.durability import TemporalDurability, worker_options
    from gwenflow.flows import Flow

dotenv.load_dotenv(override=True)

# The same flow as in demo_flow_yaml.py: durable only when it runs inside a workflow.
flow = Flow.from_file("examples/flows/news_summary.yaml", durability=TemporalDurability())


@workflow.defn
class NewsSummary:
    @workflow.run
    async def run(self) -> str:
        results = await flow.arun()  # each node runs as an activity
        return results["summarize"]


async def main():
    client = await Client.connect("localhost:7233")

    async with Worker(client, task_queue="flows", **worker_options(NewsSummary)):
        print(await client.execute_workflow(NewsSummary.run, id="news-summary-now", task_queue="flows"))

        # Every 24 hours, as long as a worker serves the "flows" task queue.
        await client.create_schedule(
            "news-summary-daily",
            Schedule(
                action=ScheduleActionStartWorkflow(NewsSummary.run, id="news-summary", task_queue="flows"),
                spec=ScheduleSpec(intervals=[ScheduleIntervalSpec(every=timedelta(days=1))]),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
