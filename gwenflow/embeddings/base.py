from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass(kw_only=True)
class Embeddings(ABC):

    model: str
    dimensions: Optional[int] = 1536

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        pass
