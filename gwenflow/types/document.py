import enum
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class DocumentCreationMode(str, enum.Enum):
    ONE_DOC_PER_FILE = "one-doc-per-file"
    ONE_DOC_PER_PAGE = "one-doc-per-page"
    ONE_DOC_PER_ELEMENT = "one-doc-per-element"


@dataclass(kw_only=True)
class Document:
    id: str | None = None
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    score: float | None = None

    def __post_init__(self) -> None:
        if self.id is None:
            self.id = hashlib.md5(self.content.encode(), usedforsecurity=False).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"id": self.id, "content": self.content, "metadata": self.metadata}
        if self.score is not None:
            result["score"] = self.score
        return result

    @classmethod
    def from_dict(cls, document: Dict[str, Any]) -> "Document":
        return cls(**document)

    @classmethod
    def from_json(cls, document: str) -> "Document":
        return cls(**json.loads(document))