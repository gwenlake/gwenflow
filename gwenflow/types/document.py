import enum
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


class DocumentCreationMode(str, enum.Enum):
    ONE_DOC_PER_FILE = "one-doc-per-file"
    ONE_DOC_PER_PAGE = "one-doc-per-page"
    ONE_DOC_PER_ELEMENT = "one-doc-per-element"


@dataclass(kw_only=True)
class Document:
    id: str | None = None
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = field(default=None)
    score: float | None = None

    def __post_init__(self) -> None:
        if self.id is None:
            self.id = hashlib.md5(self.content.encode(), usedforsecurity=False).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"id": self.id, "content": self.content, "metadata": self.metadata}
        if self.score is not None:
            result["score"] = self.score
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Document":
        return Document(**data)

    @classmethod
    def from_json(cls, json_document: str) -> "Document":
        return Document(**json.loads(json_document))