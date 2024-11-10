import context
from gwenflow.llms.openai import ChatOpenAI


messages = [
    {
        "role": "user",
        "content": "Describe Argentina in one sentence."
    }
]

# No streaming
llm = ChatOpenAI(model="gpt-4o-mini")
response = llm.chat(messages=messages)
print(response["choices"][0]["message"]["content"])

# # Streaming
stream = llm.stream(messages=messages)
for chunk in stream:
    if chunk["choices"][0]["delta"]["content"]:
        print(chunk)
