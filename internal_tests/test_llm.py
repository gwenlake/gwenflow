import context
from gwenflow import ChatOpenAI

messages = [{
    "role": "user",
    "content": "Get some recent news about Argentina and if you get a website link, visit the website"
}]

llm = ChatOpenAI(model="gpt-4o-mini")

response = llm.invoke(messages)
print(response)

print("")
stream = llm.stream(messages)
for chunk in stream:
    print(chunk)