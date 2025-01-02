import logging
import hashlib
import json
from typing import List, Optional, Dict, Any

try:
    import lancedb
    from lancedb.pydantic import Vector, LanceModel
except ImportError:
    raise ImportError("`lancedb` not installed.")

try:
    import pyarrow as pa
except ImportError:
    raise ImportError("`pyarrow` not installed.")


from gwenflow.vector_stores.base import VectorStoreBase
from gwenflow.embeddings import Embeddings, GwenlakeEmbeddings
from gwenflow.reranker import Reranker
from gwenflow.types import Document


logger = logging.getLogger(__name__)


class LanceDB(VectorStoreBase):

    def __init__(
        self,
        uri: lancedb.URI,
        collection: str = "default",
        client: Optional[lancedb.DBConnection] = None,
        embeddings: Embeddings = GwenlakeEmbeddings(),
        reranker: Optional[Reranker] = None,
        api_key: str = None,
    ):

        # Embedder
        self.embeddings = embeddings

        # reranker
        self.reranker = reranker

        # collection and uri
        self.collection = collection
        self.uri: lancedb.URI = uri
        self.table: lancedb.db.LanceTable = None

        self.client: lancedb.DBConnection = client or lancedb.connect(uri=self.uri, api_key=api_key)
        self.create()

    def create(self):
        if not self.exists():

            class schema(LanceModel):
                id: str
                vector: Vector(self.embeddings.dimensions)
                payload: str

            logger.debug(f"Creating collection: {self.collection}")
            self.table = self.client.create_table(self.collection, schema=schema, mode="overwrite", exist_ok=True)
            # self.table.create_index(column='vector', index_type='IVF_PQ', metric="cosine", num_partitions=256, num_sub_vectors=32)

        elif self.table is None:
            self.table = self.client.open_table(self.collection)


    def exists(self) -> bool:
        if self.client:
            if self.collection in self.client.table_names():
                return True
        return False
    
    def get_collections(self) -> list:
        if self.client:
            return self.client.table_names()
        return []

    def insert(self, documents: list[Document]):
        logger.info(f"Inserting {len(documents)} documents into collection {self.collection}")

        data = []
        for document in documents:
            text_for_id = document.id
            if text_for_id is None:
                text_for_id = document.content
            _id = hashlib.md5(text_for_id.encode(), usedforsecurity=False).hexdigest()
            _embeddings = self.embeddings.embed_documents([document.content])[0]
            _payload = document.metadata
            _payload["content"] = document.content
            data.append(
                dict(
                    id=_id,
                    vector=_embeddings,
                    payload=json.dumps(_payload),
                )
            )
    
        if len(data) > 0:
            for d in data:
                self.table.delete(f"id='{ d['id'] }'")
            self.table.add(data)


    def search(self, query: str, limit: int = 5, filters: dict = None) -> list[Document]:

        query_embedding = self.embeddings.embed_query(query)
        if query_embedding is None:
            logger.error(f"Error getting embedding for Query: {query}")
            return []

        results = self.table.search(
            query=query_embedding,
            vector_column_name="vector",
        ).metric("cosine").limit(limit).to_list()

        documents = []
        for item in results:
            payload = json.loads(item["payload"])
            documents.append(
                Document(
                    id=item["id"],
                    content=payload.pop("content"),
                    metadata=payload,
                    score=1-item["_distance"],
                )
            )
    
        if self.reranker:
            documents = self.reranker.rerank(query=query, documents=documents)

        return documents


    def drop(self):
        if self.exists():
            logger.debug(f"Deleting collection: {self.collection}")
            self.client.drop_table(self.collection)

    def count(self) -> int:
        if self.exists():
            return self.table.count_rows()
        return 0

    def info(self) -> dict:
        return {}
    
    def delete(self, id: int):
        return False

    def get(self, id: int) -> dict:        
        return None

    def list(self, filters: dict = None, limit: int = 100) -> list:
        return []
