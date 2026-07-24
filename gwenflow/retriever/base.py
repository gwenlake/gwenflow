import hashlib
from dataclasses import dataclass
from typing import Any, List, Optional

from gwenflow.embeddings import GwenlakeEmbeddings
from gwenflow.logger import logger
from gwenflow.parsers.text_splitters import TokenTextSplitter
from gwenflow.reranker import GwenlakeReranker
from gwenflow.telemetry import tracer
from gwenflow.types.document import Document
from gwenflow.vector_stores.base import VectorStoreBase

MIN_CONTENT_LENGTH = 20


@dataclass
class Retriever:
    name: str
    pathname: Optional[str] = None
    vector_db: Optional[VectorStoreBase] = None
    chunk_size: int = 500
    chunk_overlap: int = 100
    top_k: int = 5

    def __post_init__(self) -> None:
        if not self.vector_db:
            try:
                from gwenflow.vector_stores.lancedb import LanceDB
            except ImportError as e:
                raise ImportError("LanceDB is not installed. Please install it to use the default vector store.") from e
            try:
                uri = self.pathname or f"./{self.name}"
                self.vector_db = LanceDB(
                    uri=uri,
                    embeddings=GwenlakeEmbeddings(model="multilingual-e5-large"),
                    reranker=GwenlakeReranker(model="BAAI/bge-reranker-v2-m3"),
                )
            except Exception as e:
                logger.error(f"Error creating retriever: {e}")

    @tracer.retriever(name="Retriever Search")
    def search(self, query: str, filters: dict = None) -> list[Document]:
        try:
            if not self.vector_db:
                return []
            documents = self.vector_db.search(query, limit=10 * self.top_k, filters=filters)
            return documents[: self.top_k]
        except Exception as e:
            logger.error(f"Error searching for documents: {e}")
        return []

    def _unique_key(self, text: str) -> str:
        return hashlib.md5(text.encode(), usedforsecurity=False).hexdigest()

    def load_document(self, document: Document) -> bool:
        if not self.vector_db:
            return False
        try:
            docs = []
            text_splitter = TokenTextSplitter(
                chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap, encoding_name="cl100k_base"
            )
            for doc in text_splitter.create_documents([document.content]):
                if len(doc.page_content) > MIN_CONTENT_LENGTH:
                    docs.append(
                        Document(
                            id=self._unique_key(doc.page_content),
                            content=doc.page_content,
                        )
                    )
            if docs:
                self.vector_db.insert(docs)
            return True
        except Exception as e:
            logger.error(f"Error loading document: {e}")
        return False

    def load_documents(self, documents: List[Any]) -> bool:
        for document in documents:
            if isinstance(document, str):
                document = Document(content=document)
            self.load_document(document)
