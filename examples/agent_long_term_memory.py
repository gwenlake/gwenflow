"""Long-term memory: an agent that remembers between conversations.

The memory is a set of markdown files under ./memory/.
"""

import asyncio

import dotenv

from gwenflow import Agent, ChatOpenAI
from gwenflow.memory import LocalFileMemoryStore, MemoryConsolidator, MemoryToolset, build_memory_curator

dotenv.load_dotenv(override=True)

USER = "/memories/user/MEMORY.md"
NOTES = "/memories/thread/demo/NOTES.md"


async def main():
    llm = ChatOpenAI(model="gpt-4o-mini")
    store = LocalFileMemoryStore("./memory")
    question = "Remember that I want answers as bullet points. Also, the Orion project slips to April."

    # 1. Agent: its memory is in the prompt; it reads topic files and edits memory when asked.
    memory = MemoryToolset(
        store, always=[USER, NOTES], browse=["/memories/user/"], tools=["read_memory", "memory_edit"]
    )
    await memory.aload(query=question)
    agent = Agent(name="Assistant", llm=llm, memory=memory)
    response = await agent.arun(question)
    print(response.content)
    await memory.aflush()

    # 2. Consolidator: after the conversation, update the notes and the topic files.
    consolidator = MemoryConsolidator(llm=llm, creatable_prefixes=["/memories/user/projects/"])
    result = await consolidator.aconsolidate({NOTES: memory.snapshot()[NOTES]}, agent.history.messages)
    memory.apply(result.changed)
    await memory.aflush()

    # 3. Curator: tidy the memory (merge, deduplicate, archive).
    curator_memory = MemoryToolset(store, always=[USER, NOTES], browse=["/memories/user/"])
    await curator_memory.aload()
    await build_memory_curator(llm, curator_memory).arun("curate")
    await curator_memory.aflush()


if __name__ == "__main__":
    asyncio.run(main())
