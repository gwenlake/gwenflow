import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, model_validator, field_validator, Field, ConfigDict

from gwenflow.tools import BaseTool
from gwenflow.types.document import Document
from gwenflow.knowledge.base import Knowledge
from gwenflow.utils import logger


class KnowledgeTool(BaseTool):

    name: str = "KnowledgeTool"
    description: str = "Search in the knowledge base."

    knowledge: Optional[Knowledge] = Field(None, validate_default=True)


    @field_validator("knowledge", mode="before")
    @classmethod
    def set_knowledge(cls, v: Optional[Knowledge]) -> Knowledge:
        if not v:
            try:
                v = Knowledge()
            except Exception as e:
                logger.error(f"Error creating KnowledgeTool: {e}")
        return v

    def load_documents(self, documents: List[Any]) -> bool:
        for document in documents:
            if isinstance(document, str):
                document = Document(content=document)
            self.knowledge.load_document(document)
    
    def _run(self, query: str = Field(description="The search query.")) -> str:
        documents = self.knowledge.search(query, limit=5)
        return json.dumps([doc.to_dict() for doc in documents])
