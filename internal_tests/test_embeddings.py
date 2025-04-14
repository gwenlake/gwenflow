import context
import pandas as pd
from gwenflow.embeddings import GwenlakeEmbeddings


list_of_texts = [
    "Olympic Games will be in Paris in 2024",
    "Do Not Watch This Movie! Not funny at all",
    "Can you help me write an email to my best friend?",
]

embeddings_model = GwenlakeEmbeddings(model="e5-base-v2")
embedded_docs = embeddings_model.embed_documents(list_of_texts)
print(embedded_docs)

# embedded_query = embeddings_model.embed_query("What was the name mentioned in the conversation?")

# print(embeddings_model.model)
# print(embeddings_model.dimensions)
# print(embedded_query[:10])

