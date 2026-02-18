import hashlib
import json
from typing import Optional

try:
    import lancedb
    from lancedb.pydantic import LanceModel, Vector
except ImportError as exc:
    raise ImportError("`lancedb` not installed.") from exc


from gwenflow.embeddings import Embeddings, GwenlakeEmbeddings
from gwenflow.logger import logger
from gwenflow.reranker import Reranker
from gwenflow.types import Document
from gwenflow.vector_stores.base import VectorStoreBase


class LanceDB(VectorStoreBase):
    def __init__(
        self,
        uri: lancedb.URI,
        collection: str = "default",
        client: Optional[lancedb.DBConnection] = None,
        embeddings: Optional[Embeddings] = None,
        reranker: Optional[Reranker] = None,
        api_key: Optional[str] = None,
    ):
        """Initialize the LanceDB vector store.

        Args:
            uri (lancedb.URI): The location of the database (local path or s3://).
            collection (str, optional): Name of the table/collection. Defaults to "default".
            client (lancedb.DBConnection, optional): Existing connection instance. Defaults to None.
            embeddings (Embeddings, optional): Embedding model instance. Defaults to GwenlakeEmbeddings().
            reranker (Reranker, optional): Reranker instance for post-processing. Defaults to None.
            api_key (str, optional): API key for remote LanceDB storage (e.g., LanceDB Cloud). Defaults to None.
        """
        # Embedder
        self.embeddings = embeddings or GwenlakeEmbeddings()

        # reranker
        self.reranker = reranker

        # collection and uri
        self.collection = collection
        self.uri = uri

        # Client initialization
        self.client = client or lancedb.connect(uri=self.uri, api_key=api_key)

        self.table: Optional[lancedb.db.LanceTable] = None
        self.create()

    def create(self):
        """Create the table with the required schema if it doesn't exist."""
        if not self.exists():

            class Schema(LanceModel):
                id: str
                vector: Vector(self.embeddings.dimensions)  # type: ignore
                payload: str

            logger.debug(f"Creating collection: {self.collection}")
            self.table = self.client.create_table(self.collection, schema=Schema, mode="overwrite", exist_ok=True)
        else:
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
        logger.info(f"Embedding {len(documents)} documents")
        embeddings = self.embeddings.embed_documents([document.content for document in documents])
        logger.info(f"Inserting {len(documents)} documents into collection {self.collection}")
        data = []
        for document, embedding in zip(documents, embeddings, strict=False):
            if document.id is None:
                document.id = hashlib.md5(document.content.encode(), usedforsecurity=False).hexdigest()
            _id = document.id
            _payload = document.metadata
            _payload["content"] = document.content
            data.append(
                {
                    "id": _id,
                    "vector": embedding,
                    "payload": json.dumps(_payload),
                }
            )

        if len(data) > 0:
            for d in data:
                self.table.delete(f"id='{d['id']}'")
            self.table.add(data)

    def search(self, query: str, limit: int = 5, filters: dict = None) -> list[Document]:
        query_embedding = self.embeddings.embed_query(query)
        if query_embedding is None:
            logger.error(f"Error getting embedding for Query: {query}")
            return []

        results = (
            self.table.search(
                query=query_embedding,
                vector_column_name="vector",
            )
            .metric("cosine")
            .limit(limit)
            .to_list()
        )

        documents = []
        for item in results:
            payload = json.loads(item["payload"])
            documents.append(
                Document(
                    id=item["id"],
                    content=payload.pop("content"),
                    metadata=payload,
                    score=item["_distance"],
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
