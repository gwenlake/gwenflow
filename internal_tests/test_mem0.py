import context
import json

from gwenflow.memory.chat_memory_buffer import ChatMemoryBuffer
import os
from mem0 import Memory

m = Memory()
messages = [
    {"role": "user", "content": "I'm planning to watch a movie tonight. Any recommendations?"},
    {"role": "assistant", "content": "How about a thriller movies? They can be quite engaging."},
    {"role": "user", "content": "I'm not a big fan of thriller movies but I love sci-fi movies."},
    {"role": "assistant", "content": "Got it! I'll avoid thriller recommendations and suggest sci-fi movies in the future."}
]
result = m.add(messages, user_id="alice", metadata={"category": "movie_recommendations"})
all_memories = m.get_all(user_id="alice")
print(all_memories)
exit(0)
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