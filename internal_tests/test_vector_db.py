import context
from gwenflow import Document
from gwenflow.embeddings import GwenlakeEmbeddings
from gwenflow.reranker import GwenlakeReranker
from gwenflow.vector_stores.qdrant import Qdrant
from gwenflow.vector_stores.lancedb import LanceDB
from gwenflow.vector_stores.faiss import FAISS


client = FAISS("test_vector_db_faiss.pkl", embeddings=GwenlakeEmbeddings(model="multilingual-e5-large"))
# client.drop()

# list_of_texts = [
#     "Olympic Games will be in Paris in 2024",
#     "Do Not Watch This Movie! Not funny at all",
#     "Can you help me write an email to my best friend?",
# ]

# for i, text in enumerate(list_of_texts):
#     client.insert([Document(content=text)])

documents = client.search("email to a friend", limit=3)
for document in documents:
    print(document)

print("")

documents = client.search("sport in Paris", limit=3)
for document in documents:
    print(document)

exit(1)

# qdrant = Qdrant(collection="clinicaltrials", host="10.0.111.108", embeddings=GwenlakeEmbeddings(model="e5-base-v2"))
# documents = qdrant.search("asthma", limit=10)
# for document in documents:
#     print(document.content[:100], document.score)


# qdrant = Qdrant(collection="bciinfo", host="localhost", embeddings=GwenlakeEmbeddings(model="multilingual-e5-base"))
# documents = qdrant.search("chine", limit=10)
# for document in documents:
#     print(document.content[:100], document.score)



# qdrant = Qdrant(collection="bciinfo", host="localhost", embeddings=GwenlakeEmbeddings(model="multilingual-e5-base"), reranker=GwenlakeReranker(top_k=20))

# print("-------")
# documents = qdrant.search("des informations sur la santé en chine", limit=50)
# for document in documents:
#     print(document.content[:100], document.score)

# # print(documents[4].content)


# # # treshold
# qdrant = Qdrant(collection="bciinfo", host="localhost", embeddings=GwenlakeEmbeddings(model="multilingual-e5-base"), reranker=GwenlakeReranker(threshold=0))

# print("-------")
# documents = qdrant.search("des informations sur la santé en chine", limit=10)
# for document in documents:
#     print(document.content[:100], document.score)

# client = LanceDB(uri="./test_vector_db-lancedb", embeddings=GwenlakeEmbeddings(model="multilingual-e5-base"))


# list_of_texts = [
#     "Olympic Games will be in Paris in 2024",
#     "Do Not Watch This Movie! Not funny at all",
#     "Can you help me write an email to my best friend?",
# ]

# for i, text in enumerate(list_of_texts):
#     client.insert([Document(id=str(i), content=text)])

# print("-------")
# documents = client.search("a want to see a movie. funny", limit=10)
# for document in documents:
#     print(document.id, document.content[:100], document.score)
