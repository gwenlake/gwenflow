import context
from gwenflow import Document
from gwenflow.reranker import GwenlakeReranker


list_of_documents = [
    Document(content="Olympic Games will be in Paris in 2024"),
    Document(content="Do Not Watch This Movie! Not funny at all"),
    Document(content="Can you help me write an email to my best friend?"),
    Document(content="Something about Ping Pong and Judo."),
]

# model = GwenlakeReranker(top_k=4, threshold=-8)
# print(model.rerank_documents(query="A movie", texts=list_of_texts))

model = GwenlakeReranker(top_k=2)
print(model.rerank(query="A movie", documents=list_of_documents))
print(model.rerank(query="Something about sports", documents=list_of_documents))
# print(model.rerank_documents(query="A movie", texts=list_of_texts, parse_response=False))

# print(model.rerank_documents(query="a funny movie", texts=list_of_texts))
# print(model.rerank_documents(query="a movie not funny", texts=list_of_texts))
# print(model.rerank_documents(query="something related to sports", texts=list_of_texts))
# print(model.rerank_documents(query="something related to my friends", texts=list_of_texts))
