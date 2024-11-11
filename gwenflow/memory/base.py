
import uuid
from typing import Any
from abc import ABC

from gwenflow.types import ChatMessage


class BaseChatMemory(ABC):
 
    key: str = None
    messages: list[ChatMessage] = []

    def __init__(self, key: str = None):
        self.key = key if key else uuid.uuid4()

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

