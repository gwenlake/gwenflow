
import uuid
from typing import List, Callable, Union, Optional, Any, Dict, Iterator, Literal, Sequence, overload, Type
from pydantic import BaseModel, model_validator, field_validator, Field

from gwenflow.types import ChatMessage


class BaseChatMemory(BaseModel):
 
    id: Optional[str] = Field(None, validate_default=True)
    messages: list[ChatMessage] = []

    @field_validator("id", mode="before")
    def set_id(cls, v: Optional[str]) -> str:
        id = v or str(uuid.uuid4())
        return id

    def to_string(self) -> str:
        """Convert memory to string."""
        return self.json()

    def to_dict(self, **kwargs: Any) -> dict:
        """Convert memory to dict."""
        return self.dict()
    
    def reset(self):
        self.messages = []

    def get_all(self):
        return self.messages

    def add_messages(self, messages: list[ChatMessage]):
        for message in messages:
            self.messages.append(ChatMessage(**message))

    def add_message(self, message: ChatMessage):
        self.messages.append(ChatMessage(**message))

