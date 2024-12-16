import logging
import hashlib
from typing import List, Optional, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointIdsList,
    PointStruct,
    Range,
    VectorParams,
)

from gwenflow.vector_stores.base import VectorStoreBase
from gwenflow.embeddings import Embeddings, GwenlakeEmbeddings
from gwenflow.reranker import Reranker
from gwenflow.documents import Document


logger = logging.getLogger(__name__)


class Qdrant(VectorStoreBase):

    def __init__(
        self,
        collection: str,
        embeddings: Embeddings = GwenlakeEmbeddings(),
        distance: Distance = Distance.COSINE,
        client: QdrantClient = None,
        host: str = None,
        port: int = 6333,
        path: str = None,
        url: str = None,
        api_key: str = None,
        reranker: Optional[Reranker] = None,
    ):
        """
        Initialize the Qdrant vector store.

        Args:
            collection (str): Name of the collection.
            client (QdrantClient, optional): Existing Qdrant client instance. Defaults to None.
            host (str, optional): Host address for Qdrant server. Defaults to None.
            port (int, optional): Port for Qdrant server. Defaults to None.
            path (str, optional): Path for local Qdrant database. Defaults to None.
            url (str, optional): Full URL for Qdrant server. Defaults to None.
            api_key (str, optional): API key for Qdrant server. Defaults to None.
        """

        # Embedder
        self.embeddings = embeddings

        # Distance metric
        self.distance = distance

        # reranker
        self.reranker = reranker

        if client:
            self.client = client
        else:
            params = {}
            if api_key:
                params["api_key"] = api_key
            if url:
                params["url"] = url
            if host and port:
                params["host"] = host
                params["port"] = port
            if not params:
                params["path"] = path

            self.client = QdrantClient(**params)

        self.collection = collection
        self.create()

    def get_collections(self) -> list:
        """
        List all collections.

        Returns:
            list: List of collection names.
        """
        return self.client.get_collections()

    def create(self):
        """Create collection."""
        # Skip creating collection if already exists
        response = self.get_collections()
        for collection in response.collections:
            if collection.name == self.collection:
                logging.debug(f"Collection {self.collection} already exists. Skipping creation.")
                return

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=self.embeddings.dimensions, distance=self.distance),
        )

    def drop(self):
        """Drop collection."""
        self.client.delete_collection(collection_name=self.collection)

    def count(self) -> int:
        result = self.client.count(collection_name=self.collection, exact=True)
        return result.count

    def info(self) -> dict:
        """
        Get information about the collection.

        Returns:
            dict: Collection information.
        """
        return self.client.get_collection(collection_name=self.collection)

    def insert(self, documents: list[Document]):
        """
        Insert documents into a collection.

        Args:
            documents (list): List of documents to insert.
        """
        logger.info(f"Inserting {len(documents)} documents into collection {self.collection}")

        points = []
        for document in documents:
            _embeddings = self.embeddings.embed_documents([document.content])[0]
            _id = hashlib.md5(document.id.encode(), usedforsecurity=False).hexdigest()
            _payload = document.metadata
            _payload["content"] = document.content
            _payload["name"] = document.name
            points.append(
                PointStruct(
                    id=_id,
                    vector=_embeddings,
                    payload=_payload,
                )
            )
    
        if len(points) > 0:
            self.client.upsert(collection_name=self.collection, points=points)

    def _create_filter(self, filters: dict) -> Filter:
        """
        Create a Filter object from the provided filters.

        Args:
            filters (dict): Filters to apply.

        Returns:
            Filter: The created Filter object.
        """
        conditions = []
        for key, value in filters.items():
            if isinstance(value, dict) and "gte" in value and "lte" in value:
                conditions.append(FieldCondition(key=key, range=Range(gte=value["gte"], lte=value["lte"])))
            else:
                conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
        return Filter(must=conditions) if conditions else None

    def search(self, query: str, limit: int = 5, filters: dict = None) -> list[Document]:
        """
        Search for similar vectors.

        Args:
            query (str): Query.
            limit (int, optional): Number of results to return. Defaults to 5.
            filters (dict, optional): Filters to apply to the search. Defaults to None.

        Returns:
            list: Search results.
        """

        query_embedding = self.embeddings.embed_query(query)
        if query_embedding is None:
            logger.error(f"Error getting embedding for Query: {query}")
            return []

        query_filter = self._create_filter(filters) if filters else None
        hits = self.client.search(
            collection_name=self.collection,
            query_vector=query_embedding,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        documents = []
        for d in hits:

            if d.payload is None:
                continue

            content = None
            if "content" in d.payload:
                content = d.payload.pop("content")
            if content is None and "chunk" in d.payload:
                content = d.payload.pop("chunk")

            name = None
            if "name" in d.payload:
                name = d.payload.pop("name")

            doc = Document(id=d.id, name=name, content=content)

            doc.metadata = d.payload
            doc.score = 1 - d.score
            documents.append(doc)
    
        if self.reranker:
            documents = self.reranker.rerank(query=query, documents=documents)

        return documents

    def delete(self, id: int):
        """
        Delete a vector by ID.

        Args:
            id (int): ID of the vector to delete.
        """
        self.client.delete(
            collection_name=self.collection,
            points_selector=PointIdsList(
                points=[id],
            ),
        )

    def get(self, id: int) -> dict:
        """
        Retrieve a vector by ID.

        Args:
            id (int): ID of the vector to retrieve.

        Returns:
            dict: Retrieved vector.
        """
        result = self.client.retrieve(collection_name=self.collection, ids=[id], with_payload=True)
        return result[0] if result else None


    def list(self, filters: dict = None, limit: int = 100) -> list:
        """
        List all vectors in a collection.

        Args:
            filters (dict, optional): Filters to apply to the list. Defaults to None.
            limit (int, optional): Number of vectors to return. Defaults to 100.

        Returns:
            list: List of vectors.
        """
        query_filter = self._create_filter(filters) if filters else None
        result = self.client.scroll(
            collection_name=self.collection,
            scroll_filter=query_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return result
