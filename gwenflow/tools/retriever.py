import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, model_validator, field_validator, Field, ConfigDict

from gwenflow.logger import logger
from gwenflow.tools import BaseTool
from gwenflow.types.document import Document
from gwenflow.retriever.base import Retriever


class RetrieverTool(BaseTool):

    name: str = "RetrieverTool"
    description: str = "Use this tool for fetching documents from the knowledge base."

    retriever: Optional[Retriever] = Field(None, validate_default=True)

    @field_validator("retriever", mode="before")
    @classmethod
    def set_retriever(cls, v: Optional[Retriever]) -> Retriever:
        if not v:
            try:
                v = Retriever("default")
            except Exception as e:
                logger.error(f"Error creating RetrieverTool: {e}")
        return v

    def load_documents(self, documents: List[Any]) -> bool:
        for document in documents:
            if isinstance(document, str):
                document = Document(content=document)
            self.retriever.load_document(document)
    
    def _run(self, query: str = Field(description="The search query.")) -> str:
        documents = self.retriever.search(query)
        return json.dumps([doc.to_dict() for doc in documents])
