"""An agent described by a directory: examples/harness/customer_support/.

The directory holds everything around the model: the definition (harness.yaml), the prompt
(system_prompt.md, guardrails.md, knowledge.md, tool_guidance.md), the tools (tools/*.py),
a skill (skills/), a subagent (subagents/) and a long-term memory per customer.
The code only brings the connection, the variables and the conversation.
"""

import asyncio

import dotenv

from gwenflow import ChatOpenAI, Harness

dotenv.load_dotenv(override=True)

HARNESS = "examples/harness/customer_support"


async def main():
    # Read the files once. One agent is then built per customer, with their own variables and memory.
    harness = Harness.load(HARNESS)
    print(f"{harness.name} v{harness.config.version}: {harness.config.description}")
    print(f"  prompt parts: {', '.join(harness.prompt_parts())}")
    print(f"  tools:        {', '.join(tool.name for tool in harness.tools)}")
    print(f"  skills:       {', '.join(skill.name for skill in harness.skills)}")
    print(f"  subagents:    {', '.join(sub.name for sub in harness.subagents)}")
    print()

    # `variables` fill the `{{ name }}` of the prompt and of the memory path; the file names the
    # model, the code gives the connection it runs on.
    agent = harness.build_agent(llm=ChatOpenAI(), variables={"company": "Nordic Bikes", "customer_id": "lena"})
    print([tool.name for tool in agent.tools])  # the harness tools, the memory tools, ask_technician
    print()

    questions = [
        "Hi, I'm Lena. Where is my order NB-1057? And please remember I prefer short answers.",
        "My other bike, order NB-1042, shows E07 on the display and the motor won't start. What do I do?",
        "It works now, thanks. But I'd like to return NB-1042 anyway, it is too big for me.",
    ]
    for question in questions:
        await agent.memory.aload(query=question)  # the customer's memory, before the run
        response = await agent.arun(question)
        await agent.memory.aflush()  # what the agent wrote, after the run
        print(f"> {question}\n{response.content}\n")

    # A new conversation with the same customer: the memory is back in the prompt.
    agent = harness.build_agent(llm=ChatOpenAI(), variables={"company": "Nordic Bikes", "customer_id": "lena"})
    await agent.memory.aload()
    print("## Memory" in agent.get_system_prompt(task=""))


if __name__ == "__main__":
    asyncio.run(main())
