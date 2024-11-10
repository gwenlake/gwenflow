import context
from gwenflow.llms.openai import OpenAI
from gwenflow.llms.gwenlake import Gwenlake


messages = [
    {
        "role": "user",
        "content": "Describe Argentina in one sentence."
    }
]

# # No streaming
llm = OpenAI(model="gpt-4o-mini")
response = llm.chat(messages=messages)
print("")
print(response["choices"][0]["message"]["content"])

# # # Streaming
# stream = llm.stream(messages=messages)
# for chunk in stream:
#     if chunk["choices"][0]["delta"]["content"]:
#         print(chunk)

# No streaming
llm = Gwenlake(model="meta/llama-3.1-8b-instruct")
response = llm.chat(messages=messages)
print("")
print(response["choices"][0]["message"]["content"])

# stream = llm.stream(messages=messages)
# for chunk in stream:
#     if chunk["choices"][0]["delta"]["content"]:
#         print(chunk)
