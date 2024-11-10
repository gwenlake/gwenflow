import context
from gwenflow.llms.openai import ChatOpenAI
from gwenflow.llms.gwenlake import ChatGwenlake


messages = [
    {
        "role": "user",
        "content": "Describe Argentina in one sentence."
    }
]

# # No streaming
llm = ChatOpenAI(model="gpt-4o-mini")
response = llm.chat(messages=messages)
print(response["choices"][0]["message"]["content"])

# # # Streaming
# stream = llm.stream(messages=messages)
# for chunk in stream:
#     if chunk["choices"][0]["delta"]["content"]:
#         print(chunk)

# No streaming
llm = ChatGwenlake(model="meta/llama-3.1-8b-instruct")
response = llm.chat(messages=messages)
print(response["choices"][0]["message"]["content"])

# stream = llm.stream(messages=messages)
# for chunk in stream:
#     if chunk["choices"][0]["delta"]["content"]:
#         print(chunk)
