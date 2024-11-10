import context
from gwenflow.memory.chat_memory_buffer import ChatMemoryBuffer


messages = [
    {
        "role": "user",
        "content": "anything about alzheimer?"
    },
    {
        "role": "assitant",
        "content": "Alzheimer is a disease."
    },
]

memory = ChatMemoryBuffer(token_limit=20)
memory.add_messages(messages)
print("")
print(memory.get())

memory.add_message({"role": "user", "content": "anything else you may add?"})
memory.add_message({"role": "assistant", "content": "No"})
memory.add_messages([{"role": "user", "content": "are you sure?"}])
memory.add_message({"role": "assistant", "content": "Yes, nothing to add!"})
print("")
print(memory.get())
print(memory.key)