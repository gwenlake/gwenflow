import logging
import hashlib
from typing import Optional
from enum import Enum

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointIdsList,
    PointStruct,
    Range,
    DatetimeRange,
    VectorParams,
    CreateAliasOperation,
    CreateAlias,
    DeleteAliasOperation,
    DeleteAlias,
    PayloadSchemaType)

from gwenflow.vector_stores.base import VectorStoreBase
from gwenflow.embeddings import Embeddings, GwenlakeEmbeddings
from gwenflow.reranker import Reranker
from gwenflow.types import Document

logger = logging.getLogger(__name__)

class Index(str, Enum):
    keyword = PayloadSchemaType.KEYWORD
    integer = PayloadSchemaType.INTEGER
    float = PayloadSchemaType.FLOAT
    bool = PayloadSchemaType.BOOL
    geo = PayloadSchemaType.GEO
    datetime = PayloadSchemaType.DATETIME
    text = PayloadSchemaType.TEXT

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
        on_disk: bool = True,
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

        # on disk:
        self.on_disk = on_disk

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
            # if self.on_disk:
            #     params = { "url": "http://localhost:6333" }

            self.client = QdrantClient(**params)

        self.collection = collection
        self.create()

    def get_collections(self) -> list:
        """
        List all collections.

        Returns:
            list: List of collection names.
        """
        try:
            return self.client.get_collections()
        except Exception as e:
            logger.error(f"Error while reading collections: {e}")
        return []

    def get_aliases(self) -> list:
            """
            List all aliases.

            Returns:
                list: List of alias names.
            """
            try:
                return self.client.get_aliases()
            except Exception as e:
                logger.error(f"Error while reading aliases: {e}")
            return []


    def create(self):
        """Create collection."""
        # Skip creating collection if already exists or an existing alias is already present
        collections = self.get_collections()
        if collections:
            for collection in collections.collections:
                if collection.name == self.collection:
                    logging.debug(f"Collection {self.collection} already exists. Skipping creation.")
                    return

        aliases = self.get_aliases()
        if aliases:
            for alias in aliases.aliases:
                if alias.alias_name == self.collection:
                    logging.debug(f"Alias {self.collection} already exists. Skipping creation.")
                    return
        try:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.embeddings.dimensions, distance=self.distance, on_disk=self.on_disk),
            )
        except Exception as e:
            logger.error(f"Error while creating collection: {e}")

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
        _embeddings = self.embeddings.embed_documents([document.content for document in documents])
        points = []
        for document, embedding in zip(documents, _embeddings):
            if document.id is None:
                document.id = hashlib.md5(document.content.encode(), usedforsecurity=False).hexdigest()
            _id = document.id
            _payload = document.metadata
            _payload["content"] = document.content
            points.append(
                PointStruct(
                    id=_id,
                    vector=embedding,
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
        collection_payload = self.client.get_collection(collection_name=self.collection).payload_schema
        for key, value in filters.items():
            if isinstance(value, dict) and "gte" in value and "lte" in value:
                if key in collection_payload and collection_payload[key].data_type=="datetime":
                    conditions.append(FieldCondition(key=key, range=DatetimeRange(gte=value["gte"], lte=value["lte"])))
                else:
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
            elif "chunk" in d.payload:
                content = d.payload.pop("chunk")

            documents.append(
                Document(
                    id=d.id,
                    content=content,
                    metadata=d.payload,
                    score=1-d.score,
                )
            )
    
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
    
    def create_alias(self, alias_name: str):
        """
        Create an alias for a list of collections.

        Args:
            alias (str): Name of the alias.
            collections (List[str]): List of collection names.
        """
        self.client.update_collection_aliases(
        change_aliases_operations=[
            CreateAliasOperation(
                create_alias=CreateAlias(
                    collection_name=self.collection, alias_name=alias_name
                )
            )
        ]
    )
        
    def delete_alias(self, alias_name: str):
        """
        Delete an alias.

        Args:
            alias (str): Name of the alias.
        """
        self.client.update_collection_aliases(
        change_aliases_operations=[
            DeleteAliasOperation(
                delete_alias=DeleteAlias(alias_name=alias_name)
            ),
        ]
    )
        
    def switch_alias(self, alias_name: str):
        """
        Switch an alias to a new collection.

        Args:
            alias (str): Name of the alias.
            collection (str): Name of the collection.
        """
        self.client.update_collection_aliases(
        change_aliases_operations=[
            DeleteAliasOperation(
                delete_alias=DeleteAlias(alias_name=alias_name)
            ),
            CreateAliasOperation(
                create_alias=CreateAlias(
                    collection_name=self.collection, alias_name=alias_name
                )
            )
        ]
    )
        
    def add_index(self, field_name: str, index_type: str):
        """
        Args:
            field_name (str): Name of the field to index.
            index_type (str): Type of index (must be one of the valid types).

        Raises:
            ValueError: If the index type is not valid.
        """

        if index_type not in Index.__members__:
            raise ValueError(f"Invalid index_type: {index_type}. Must be one of {list(Index.__members__.keys())}")
        
        self.client.create_payload_index(
            collection_name=self.collection,
            field_name=field_name,
            field_schema=Index(index_type)
        )
         