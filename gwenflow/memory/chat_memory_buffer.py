from typing import Any, Callable, Dict, List, Optional

from gwenflow.utils.tokens import num_tokens_from_string
from gwenflow.base.types import ChatMessage
from gwenflow.memory.base import BaseChatMemory


DEFAULT_TOKEN_LIMIT_RATIO = 0.75
DEFAULT_TOKEN_LIMIT = 3000


class ChatMemoryBuffer(BaseChatMemory):
 
    token_limit: int

    def __init__(self, token_limit, key: str = None):
        super().__init__(key)
        self.token_limit = token_limit
        self.tokenizer_fn = num_tokens_from_string

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
