import io
import os
import logging
import hashlib
import numpy as np
import pickle
from typing import Optional, Any

from gwenflow.utils.aws import aws_s3_download_in_buffer, aws_s3_is_file, aws_s3_upload_fileobj, aws_s3_uri_to_bucket_key

try:
    import faiss
except ImportError:
    raise ImportError("`faiss` is not installed.")

from gwenflow.logger import logger
from gwenflow.vector_stores.base import VectorStoreBase
from gwenflow.embeddings import Embeddings, GwenlakeEmbeddings
from gwenflow.reranker import Reranker
from gwenflow.types import Document



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
        if self.filename.startswith('s3://'):
            bucket, key = aws_s3_uri_to_bucket_key(self.filename)
            return aws_s3_is_file(bucket=bucket, key=key)      
        else:
            return os.path.isfile(self.filename)
    
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
            if(idx==-1):
                continue
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
            if self.filename.startswith('s3://'):
                logger.info(f"Saving FAISS index to S3 at {self.filename} ...")
                bucket, key = aws_s3_uri_to_bucket_key(self.filename)
                buffer = io.BytesIO()
                pickle.dump(faiss_data, buffer)
                buffer.seek(0)
                aws_s3_upload_fileobj(bucket=bucket, key=key, fileobj=buffer)
            else:
                logger.info(f"Saving FAISS index locally at {self.filename} ...")
                with open(self.filename, "wb") as f:
                    pickle.dump(faiss_data, f)
            return True
        except Exception as e:
            logger.error(e)
        return False

    def load(self):
        try:
            logger.info(f"Loading FAISS index from: {self.filename} ...")
            if self.filename.startswith("s3://"):
                bucket, key = aws_s3_uri_to_bucket_key(self.filename)
                buffer = aws_s3_download_in_buffer(bucket=bucket, key=key)
                faiss_data = pickle.load(buffer)
            else:
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
    
    def delete_files(self, ids: list[str]) -> bool:
        logger.info(f"Deleting file with document_id: {ids}")

        keep_indices = [
            i for i, doc in enumerate(self.metadata)
            if doc.get("metadata", {}).get("document_id") not in ids
        ]

        if len(keep_indices) == len(self.metadata):
            logger.warning(f"No chunks found for document_id: {ids}")
            return False

        new_index = faiss.IndexFlatL2(self.embeddings.dimensions)
        new_metadata = []

        for i in keep_indices:
            vector = self.index.reconstruct(i)
            new_index.add(np.array([vector], dtype="float32"))
            new_metadata.append(self.metadata[i])

        self.index = new_index
        self.metadata = new_metadata
        self.save()

        return True

    def get(self, id: int) -> dict:        
        return None

    def list(self, filters: dict = None, limit: int = 100) -> list:
        return []