import os
import logging
import hashlib
import numpy as np
import pickle
from typing import Optional, Any

try:
    import faiss
except ImportError:
    raise ImportError("`faiss` is not installed.")

from gwenflow.vector_stores.base import VectorStoreBase
from gwenflow.embeddings import Embeddings, GwenlakeEmbeddings
from gwenflow.reranker import Reranker
from gwenflow.types import Document


logger = logging.getLogger(__name__)


class FAISS(VectorStoreBase):

    def __init__(
        self,
        filename: str,
        embeddings: Embeddings = GwenlakeEmbeddings(),
        reranker: Optional[Reranker] = None,
    ):

        # Embedder
        self.embeddings = embeddings

        # reranker
        self.reranker = reranker

        # name
        self.filename = filename
        if not self.filename.endswith(".pkl"):
            self.filename = self.filename + ".pkl"

        self.index = None
        self.metadata = []
        self.create()

    def create(self):
        if not self.exists():
            self.index = faiss.IndexFlatL2(self.embeddings.dimensions)
            self.metadata = []
            self.save()
        else:
            self.load()

    def exists(self) -> bool:
        if os.path.isfile(self.filename):
            return True
        return False
    
    def get_collections(self) -> list:
        return []

    def insert(self, documents: list[Document]):
        logger.info(f"Embedding {len(documents)} documents")
        embeddings = self.embeddings.embed_documents([document.content for document in documents])
        embeddings = np.array(embeddings, dtype='float32')

        logger.info(f"Inserting {len(documents)} documents into index")
        data = []
        for document in documents:
            if document.id is None:
                document.id = hashlib.md5(document.content.encode(), usedforsecurity=False).hexdigest()
            data.append(document.model_dump())
    
        if len(documents) > 0:
            self.index.add(embeddings)
            self.metadata.extend(data)
            self.save()

    def search(self, query: str, limit: int = 5) -> list[Document]:

        if not self.index:
            logger.error(f"Error no index.")
            return []

        query_embedding = self.embeddings.embed_query(query)
        if query_embedding is None:
            logger.error(f"Error getting embedding for Query: {query}")
            return []
        query_embedding = np.array([query_embedding], dtype='float32')
        
        D, I = self.index.search(query_embedding, k=limit)

        documents = []
        for idx, score in zip(I[0], D[0]):
            document = self.metadata[idx].copy()
            document.pop("embedding")
            document = Document(**document)
            document.score = score
            documents.append(document)
    
        if self.reranker:
            documents = self.reranker.rerank(query=query, documents=documents)

        return documents

    def save(self):
        try:
            faiss_data = dict(index=self.index, metadata=self.metadata)
            with open(self.filename, "wb") as f:
                pickle.dump(faiss_data, f)
            return True
        except Exception as e:
            logger.error(e)
        return False

    def load(self):
        try:
            with open(self.filename, "rb") as f:
                faiss_data = pickle.load(f)
                self.index = faiss_data["index"]
                self.metadata = faiss_data["metadata"]
        except Exception as e:
            logger.error(e)
    
    def drop(self):
        try:
            os.remove(self.filename)
            self.index = faiss.IndexFlatL2(self.embeddings.dimensions)
            self.metadata = []
            return True
        except Exception as e:
            logger.error(e)
        return False

    def count(self) -> int:
        return 0

    def info(self) -> dict:
        return {}
    
    def delete(self, id: int):
        return False

    def get(self, id: int) -> dict:        
        return None

    def list(self, filters: dict = None, limit: int = 100) -> list:
        return []
