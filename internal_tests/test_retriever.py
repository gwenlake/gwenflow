import context
import json
from gwenflow import Document, Retriever
from gwenflow.readers import WebsiteReader
from gwenflow import set_log_level_to_debug

set_log_level_to_debug()


list_of_texts = [
    "Olympic Games will be in Paris in 2024",
    "Do Not Watch This Movie! Not funny at all",
    "Can you help me write an email to my best friend?",
]

retriever = Retriever(name="test")
retriever.load_documents(list_of_texts)


print("-------")
documents = retriever.search("I want to see a funny movie.")
for document in documents:
    print(document.content[:100], document.score)

print("-------")
documents = retriever.search("I want to see a movie about friendship.")
for document in documents:
    print(document.content[:100], document.score)



retriever = Retriever(name="gwenlake-website")
reader = WebsiteReader(delay=False)
documents = reader.read("https://gwenlake.com")
retriever.load_documents(documents)
# for d in documents:
#     print(json.dumps(d.model_dump(), indent=4))

print("-------")
documents = retriever.search("Can you tell me something about Gwenlake?")
for document in documents:
    print(document.content[:100], document.score)
