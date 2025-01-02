from typing import List, Optional

from pydantic import BaseModel, ConfigDict
from gwenflow.types import Document



class Reranker(BaseModel):
    """Base class for rerankers"""

    model: str
    top_k: Optional[int] = None
    threshold: Optional[float] = None

    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        raise NotImplementedError
    