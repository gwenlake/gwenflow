
from pydantic import BaseModel, Field
from typing import Any, Dict, Optional, Union
from collections.abc import Sequence


class ChatMessage(BaseModel):
    """Chat message class."""

    content: Union[str, list[Union[str, dict]]]
    role: str
    name: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict, hash=False)

    def __str__(self) -> str:
        return f"{self.role}: {self.content}"

    def to_dict(self, **kwargs: Any) -> Dict[str, Any]:
        return self.model_dump(**kwargs)
    
    def to_openai(self) -> Dict[str, Any]:
        return { "role": self.role, "content": self.content }


def messages_to_dict(messages: Sequence[ChatMessage]) -> list[dict]:
    """Convert a sequence of Messages to a list of dictionaries.

    Args:
        messages: Sequence of messages (as ChatMessages) to convert.

    Returns:
        List of messages as dicts.
    """
    return [m.to_dict() for m in messages]


def messages_to_openai(messages: Sequence[ChatMessage]) -> list[dict]:
    """Convert a sequence of Messages to a list of dictionaries in OpenAI format.

    Args:
        messages: Sequence of messages (as ChatMessages) to convert.

    Returns:
        List of messages as dicts.
    """
    return [m.to_openai() for m in messages]
