import uuid
from dataclasses import dataclass, field
from typing import Callable

from gwenflow.types import Message
from gwenflow.utils.tokens import num_tokens_from_string


@dataclass
class BaseChatMemory:
    id: str | None = None
    system_prompt: str | None = None
    messages: list[Message] = field(default_factory=list)
    tokenizer_fn: Callable | None = None

    def __post_init__(self) -> None:
        if self.id is None:
            self.id = str(uuid.uuid4())
        if self.tokenizer_fn is None:
            self.tokenizer_fn = num_tokens_from_string

    def reset(self):
        self.messages = []

    def get_all(self):
        messages = []
        if self.system_prompt:
            messages.append(Message(role="system", content=self.system_prompt))
        messages.extend(self.messages)
        return messages

    def add_message(self, message):
        if isinstance(message, Message):
            self.messages.append(message)
        elif isinstance(message, dict):
            self.messages.append(Message(**message))
        else:
            self.messages.append(Message(**message.__dict__))

    def add_messages(self, messages: list[Message]):
        for message in messages:
            self.add_message(message)

    def _token_count_for_messages(self, messages: list[Message]) -> int:
        if not messages:
            return 0
        text = " ".join(str(m.content) for m in messages)
        return self.tokenizer_fn(text)
