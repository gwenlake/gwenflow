
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, model_validator, field_validator, Field, ConfigDict

import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter

from gwenflow.types.document import Document
from gwenflow.embeddings import GwenlakeEmbeddings
# from gwenflow.vector_stores.base import VectorStoreBase
from gwenflow.vector_stores.qdrant import Qdrant
from gwenflow.utils import logger


class Knowledge(BaseModel):

    vector_db: Optional[Qdrant] = None

    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True, extra="allow")

    def search(self, query: str, limit: int = 5, filters: dict = None) -> list[Document]:
        try:
            if not self.vector_db:
                return []
            return self.vector_db.search(query, limit=limit, filters=filters)
        except Exception as e:
            logger.error(f"Error searching for documents: {e}")
        return []

    def _create_vector_db(self):
        if not self.vector_db:
            try:
                collection_name = uuid.uuid4().hex
                self.vector_db = Qdrant(collection=collection_name, embeddings=GwenlakeEmbeddings(model="multilingual-e5-base"), on_disk=False)
            except Exception as e:
                logger.error(f"Error creating knowledge: {e}")

    def load_document(self, document: Document) -> bool:
        try:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
            docs = [Document(content=p.page_content) for p in text_splitter.create_documents([document.content])]
            if len(docs)>0:
                self._create_vector_db()
                if not self.vector_db:
                    return False
                self.vector_db.insert(docs)
            return True
        except Exception as e:
            logger.error(f"Error loading document: {e}")
        return False
    
