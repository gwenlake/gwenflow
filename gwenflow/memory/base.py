
import uuid
from typing import Optional, Callable, Any, List
from pydantic import BaseModel, field_validator, Field

from gwenflow.types import Message
from gwenflow.utils.tokens import num_tokens_from_string


class BaseChatMemory(BaseModel):
 
    id: Optional[str] = Field(None, validate_default=True)
    system_prompt: Optional[str] = None
    messages: list[Message] = []
    tokenizer_fn: Optional[Callable] = Field(None, validate_default=True)

    @field_validator("id", mode="before")
    def set_id(cls, v: Optional[str]) -> str:
        id = v or str(uuid.uuid4())
        return id

    @field_validator("tokenizer_fn", mode="before")
    def set_tokenizer_fn(cls, v: Optional[Callable]) -> Callable:
        fn = v or num_tokens_from_string
        return fn
    
    def to_string(self) -> str:
        """Convert memory to string."""
        return self.json()

    def to_dict(self, **kwargs: Any) -> dict:
        """Convert memory to dict."""
        return self.dict()
    
    def reset(self):
        self.messages = []

    def get_all(self):
        messages = []
        if self.system_prompt:
            messages.append(Message(role="system", content=self.system_prompt))
        messages.extend(self.messages)
        return messages

    def add_message(self, message: Message):
        if isinstance(message, Message):
            self.messages.append(message)
        elif isinstance(message, dict):
            self.messages.append(Message(**message))
        else:
            self.messages.append(Message(**message.__dict__))

    def add_messages(self, messages: list[Message]):
        for message in messages:
            self.add_message(message)

    def _token_count_for_messages(self, messages: List[Message]) -> int:
        if len(messages) <= 0:
            return 0
        text = " ".join(str(m.content) for m in messages)
        return self.tokenizer_fn(text)
