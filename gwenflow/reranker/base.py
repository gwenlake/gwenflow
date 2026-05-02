from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from gwenflow.types import Document


@dataclass(kw_only=True)
class Reranker(ABC):

    model: str
    top_k: Optional[int] = None
    threshold: Optional[float] = None

    @abstractmethod
    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        pass
