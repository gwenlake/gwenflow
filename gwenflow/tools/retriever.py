from dataclasses import dataclass
from typing import Any, List, Optional

from pydantic import Field

from gwenflow.logger import logger
from gwenflow.retriever.base import Retriever
from gwenflow.tools.tool import Tool
from gwenflow.types.document import Document


@dataclass(kw_only=True)
class RetrieverTool(Tool):
    name: str = "RetrieverTool"
    description: str = "Use this tool for fetching documents from the knowledge base."
    retriever: Optional[Retriever] = None

    def __post_init__(self) -> None:
        if not self.retriever:
            try:
                self.retriever = Retriever(name="default")
            except Exception as e:
                logger.error(f"Error creating RetrieverTool: {e}")
        super().__post_init__()

    def load_documents(self, documents: List[Any]) -> bool:
        for document in documents:
            if isinstance(document, str):
                document = Document(content=document)
            self.retriever.load_document(document)

    def _run(self, query: str = Field(description="The search query.")):
        documents = self.retriever.search(query)
        return [doc.to_dict() for doc in documents]
