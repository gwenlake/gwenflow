
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, model_validator, field_validator, Field, ConfigDict

import hashlib
import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter

from gwenflow.types.document import Document
from gwenflow.embeddings import GwenlakeEmbeddings
from gwenflow.vector_stores.base import VectorStoreBase
from gwenflow.vector_stores.lancedb import LanceDB
from gwenflow.utils import logger


MIN_CONTENT_LENGTH = 20


class Knowledge(BaseModel):

    name: str

    uri: Optional[str] = None
    vector_db: Optional[VectorStoreBase] = None
    chunk_size: Optional[int] = 500
    chunk_overlap: Optional[int] = 100

    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)
   
    @model_validator(mode="after")
    def model_valid(self) -> Any:
        if not self.vector_db:
            try:
                uri = self.uri or f"./{ self.name }"
                self.vector_db = LanceDB(uri=uri, embeddings=GwenlakeEmbeddings(model="multilingual-e5-large"))
            except Exception as e:
                logger.error(f"Error creating knowledge: {e}")
        return self
    
    def search(self, query: str, limit: int = 5, filters: dict = None) -> list[Document]:
        try:
            if not self.vector_db:
                return []
            return self.vector_db.search(query, limit=limit, filters=filters)
        except Exception as e:
            logger.error(f"Error searching for documents: {e}")
        return []

    def _unique_key(self, text: str):
        return hashlib.md5(text.encode(), usedforsecurity=False).hexdigest()
    
    def load_document(self, document: Document) -> bool:
        if not self.vector_db:
            return False

        try:
            docs = []
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)

            for doc in text_splitter.create_documents([document.content]):
                if len(doc.page_content) > MIN_CONTENT_LENGTH:
                    docs.append(
                        Document(
                            id=self._unique_key(doc.page_content), # id based on content to remove content duplicates in knowledge
                            content=doc.page_content,
                        )
                    )

            if len(docs)>0:
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
    