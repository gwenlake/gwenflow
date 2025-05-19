from gwenflow import Agent, ChatOpenAI
from gwenflow.memory.chat_memory_buffer import ChatMemoryBuffer


messages = [
    {
        "role": "user",
        "content": "anything about alzheimer?"
    },
    {
        "role": "assistant",
        "content": "Alzheimer is a disease."
    },
]

memory = ChatMemoryBuffer(token_limit=300)
memory.system_prompt = "My system"
memory.add_messages(messages)
memory.add_message({"role": "user", "content": "anything else you may add?"})
memory.add_message({"role": "assistant", "content": "No"})
memory.add_messages([{"role": "user", "content": "are you sure?"}])
memory.add_message({"role": "assistant", "content": "Yes, nothing to add!"})
memory.add_messages([{"role": "user", "content": "are you sure?"}])
memory.add_message({"role": "assistant", "content": "Yes nothing to add !"})
memory.add_message({"role": "user", "content": "Can you explain Parkinson to me ?"})
memory.add_message({"role": "assistant", "content": "It is also a disease"})
memory.add_messages([{"role": "user", "content": "Can you describe it to me ?"}])
memory.add_message({"role": "assistant", "content": "A progressive disorder affecting the nervous system, causing tremors, muscle rigidity, and motor difficulties."})

agent = Agent(
    name="Medical assistant",
    llm=ChatOpenAI(model="gpt-4o-mini"),
    history=memory
)

response = agent.run_stream("What were we talking about at first ?")
for chunk in response:
    if chunk.thinking:
        print(chunk.thinking)
    if chunk.content:
        print(chunk.content, end="")