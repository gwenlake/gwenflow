import context
import json

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
memory.system_prompt = "mon systeme"
memory.add_messages(messages)
# print("")
# print(memory.get())
# exit(1)

memory.add_message({"role": "user", "content": "anything else you may add?"})
memory.add_message({"role": "assistant", "content": "No"})
memory.add_messages([{"role": "user", "content": "are you sure?"}])
memory.add_message({"role": "assistant", "content": "Yes, nothing to add!"})
memory.add_messages([{"role": "user", "content": "are you sure?"}])
memory.add_message({"role": "assistant", "content": "Start. Yes, nothing to add! Yes, nothing to add! Yes, nothing to add! Yes, nothing to add! Yes, nothing to add! Yes, nothing to add! Yes, nothing to add! End."})
print("")
print(memory.get())

# print("")
# print("Generated Key:", memory.id)