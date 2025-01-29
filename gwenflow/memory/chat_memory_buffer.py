from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel, model_validator, field_validator, Field

from gwenflow.utils.tokens import num_tokens_from_string
from gwenflow.types import ChatMessage
from gwenflow.memory.base import BaseChatMemory


DEFAULT_TOKEN_LIMIT = 8192
DEFAULT_TOKEN_LIMIT_RATIO = 0.75


class ChatMemoryBuffer(BaseChatMemory):
 
    token_limit: Optional[int] = Field(None, validate_default=True)
    tokenizer_fn: Optional[Callable] = Field(None, validate_default=True)

    @field_validator("token_limit", mode="before")
    def set_token_limit(cls, v: Optional[int]) -> int:
        token_limit = v or int(DEFAULT_TOKEN_LIMIT * DEFAULT_TOKEN_LIMIT_RATIO)
        return token_limit
    
    @field_validator("tokenizer_fn", mode="before")
    def set_tokenizer_fn(cls, v: Optional[Callable]) -> Callable:
        fn = v or num_tokens_from_string
        return fn
    
    def get(self, initial_token_count: int = 0):

        chat_history = self.get_all()

        if initial_token_count > self.token_limit:
            raise ValueError("Initial token count exceeds token limit")
    
        message_count = len(chat_history)

        cur_messages = chat_history[-message_count:]
        token_count = self._token_count_for_messages(cur_messages) + initial_token_count

        while token_count > self.token_limit and message_count > 1:
            message_count -= 1
            while chat_history[-message_count].role in (
                "tool",
                "assistant",
            ):
                message_count -= 1

            cur_messages = chat_history[-message_count:]
            token_count = (
                self._token_count_for_messages(cur_messages) + initial_token_count
            )

        # catch one message longer than token limit
        if token_count > self.token_limit or message_count <= 0:
            return []

        return chat_history[-message_count:]

    def _token_count_for_messages(self, messages: List[ChatMessage]) -> int:
        if len(messages) <= 0:
            return 0
        text = " ".join(str(m.content) for m in messages)
        return self.tokenizer_fn(text)
