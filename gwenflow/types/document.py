
import uuid
from typing import Optional, Any, Dict
from pydantic import BaseModel, ConfigDict, field_validator, Field


class Document(BaseModel):
    """Base class for Documents"""

    id: Optional[str] = Field(None, validate_default=True)
    content: str
    metadata: Dict[str, Any] = {}
    score: Optional[float] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("id", mode="before")
    def set_id(cls, v: Optional[str]) -> str:
        id = v or str(uuid.uuid4())
        return id

    def to_dict(self) -> Dict[str, Any]:
        """Returns a dictionary representation of the document"""
        return self.model_dump(include={"id", "content", "metadata", "score"}, exclude_none=True)

    @classmethod
    def from_dict(cls, document: Dict[str, Any]) -> "Document":
        """Returns a Document object from a dictionary representation"""
        return cls.model_validate(**document)

    @classmethod
    def from_json(cls, document: str) -> "Document":
        """Returns a Document object from a json string representation"""
        return cls.model_validate_json(document)