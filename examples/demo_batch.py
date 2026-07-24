# import dotenv

# from gwenflow import ChatOpenAI

# dotenv.load_dotenv(override=True)

# llm = ChatOpenAI(model="gpt-4o-mini")

# questions = {
#     "capital-australia": "What is the capital of Australia?",
#     "www-inventor": "Who invented the World Wide Web?",
#     "python-latest": "In one sentence, what is new in the latest version of Python?",
# }

# print("Submitting batch...")
# batch = llm.create_batch(
#     inputs=list(questions.values()),
#     custom_ids=list(questions.keys()),
#     metadata={"purpose": "gwenflow batch example"},
# )
# print(f"Batch {batch.id} created, status={batch.status}")

# print("Polling until the batch finishes (this can take a while)...")
# batch = llm.poll_batch(batch.id, poll_interval=10)
# print(f"Batch {batch.id} finished with status={batch.status}")

# results = llm.get_batch_results(batch.id)
# for custom_id, question in questions.items():
#     item = results[custom_id]
#     print(f"\nQ [{custom_id}]: {question}")
#     if item.error:
#         print(f"A: error — {item.error}")
#     else:
#         print(f"A: {item.response.content}")


from gwenflow import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")
batch = llm.retrieve_batch("batch_6a632ab53c848190a8b170b1676e75aa")
print(batch.status)
