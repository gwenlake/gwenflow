from typing import Optional
from pydantic import field_validator, Field

from gwenflow.types import Message
from gwenflow.memory.base import BaseChatMemory
from gwenflow.logger import logger


DEFAULT_TOKEN_LIMIT = 8192
DEFAULT_TOKEN_LIMIT_RATIO = 0.75


class ChatMemoryBuffer(BaseChatMemory):
 
    token_limit: Optional[int] = Field(None, validate_default=True)

    @field_validator("token_limit", mode="before")
    def set_token_limit(cls, v: Optional[int]) -> int:
        token_limit = v or int(DEFAULT_TOKEN_LIMIT * DEFAULT_TOKEN_LIMIT_RATIO)
        return token_limit
    
    def get(self):

        initial_token_count = 0

        if self.system_prompt:
            initial_token_count = self._token_count_for_messages([Message(role="system", content=self.system_prompt)])
        if initial_token_count > self.token_limit:
            raise ValueError("Initial token count exceeds token limit")

        chat_history = self.messages
        message_count = len(chat_history)

        cur_messages = chat_history[-message_count:]
        token_count = self._token_count_for_messages(cur_messages) + initial_token_count

        while token_count > self.token_limit and message_count > 1:
            message_count -= 1
            while chat_history[-message_count].role in ("tool", "assistant"):
                message_count -= 1
            cur_messages = chat_history[-message_count:]
            token_count = self._token_count_for_messages(cur_messages) + initial_token_count

        if token_count > self.token_limit and not self.system_prompt:
            logger.warning("Token limit exceeded.")
            return []
 
        if self.system_prompt:
            return [Message(role="system", content=self.system_prompt)] + chat_history[-message_count:]
        
        return chat_history[-message_count:]
