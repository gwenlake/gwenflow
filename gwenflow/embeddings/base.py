from typing import Optional, List

from pydantic import BaseModel, ConfigDict


class Embeddings(BaseModel):
    """Base class for embeddings"""

    model: str
    dimensions: Optional[int] = 1536

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> List[float]:
        raise NotImplementedError
