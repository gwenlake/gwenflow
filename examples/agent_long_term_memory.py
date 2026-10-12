"""Long-term markdown memory on real files.

Run it twice. The first run records a preference in MEMORY.md and a topic file
under ./memory/; the second run starts with MEMORY.md in the prompt and the
topic file recalled because the question names it.
"""

import asyncio

import dotenv

from gwenflow import Agent, ChatOpenAI
from gwenflow.memory import LocalFileMemoryStore, MemoryToolset

dotenv.load_dotenv(override=True)


async def main() -> None:
    task = (
        "Remember that I want short answers. Also keep a note that the Orion project (ref 42) "
        "is owned by Alice and ships in March. Then tell me what you know about Orion."
    )
    store = LocalFileMemoryStore("./memory")
    memory = MemoryToolset(store, always=["/memories/user/MEMORY.md"], browse=["/memories/user/"])
    await memory.aload(query=task)

    agent = Agent(name="Assistant", llm=ChatOpenAI(model="gpt-4o-mini"), memory=memory)
    async for event in agent.arun_stream(task):
        if getattr(event, "content", None):
            print(event.content, end="")
    print()

    report = await memory.aflush()
    print(f"written: {[f.path for f in report.written]}, conflicts: {report.conflicts}")


if __name__ == "__main__":
    asyncio.run(main())
