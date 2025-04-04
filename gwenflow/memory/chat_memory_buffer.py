from typing import Optional
from pydantic import field_validator, Field

from gwenflow.types import Message
from gwenflow.memory.base import BaseChatMemory


DEFAULT_TOKEN_LIMIT = 8192
DEFAULT_TOKEN_LIMIT_RATIO = 0.75


class ChatMemoryBuffer(BaseChatMemory):
 
    token_limit: Optional[int] = Field(None, validate_default=True)

    @field_validator("token_limit", mode="before")
    def set_token_limit(cls, v: Optional[int]) -> int:
        token_limit = v or int(DEFAULT_TOKEN_LIMIT * DEFAULT_TOKEN_LIMIT_RATIO)
        return token_limit
    
    def keep_tokens_from_text(self, text: str, token_limit= int):
        num_tokens = 0
        truncated_text = []
        for token in text.split(" "):
            num_tokens = self.tokenizer_fn(" ".join(truncated_text))
            if num_tokens > token_limit:
                break
            truncated_text.append(token)
        return " ".join(truncated_text)

    def _get_truncated_messages(self, messages: list):

        chat_history = []
        token_count = 0

        if self.system_prompt:
            token_count = self._token_count_for_messages([Message(role="system", content=self.system_prompt)])
        if token_count > self.token_limit:
            raise ValueError("Initial token count exceeds token limit")

        last_user_message = False
        last_assistant_message = False

        for message in reversed(messages):

            if not last_user_message and message.role == "user":
                chat_history.append(message)
                last_user_message = True

            if not last_assistant_message and message.role == "assistant":
                chat_history.append(message)
                last_assistant_message = True

            if last_user_message and last_assistant_message:
                break

        if self.system_prompt:
            chat_history.append(Message(role="system", content=self.system_prompt))

        chat_history = list(reversed(chat_history))

        token_count = self._token_count_for_messages(chat_history)
        
        # truncate last message
        if token_count > self.token_limit:
            token_limit = self.token_limit - self._token_count_for_messages(chat_history[:-1])
            chat_history[-1].content = self.keep_tokens_from_text(chat_history[-1].content, token_limit=token_limit)

        return chat_history


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

        # we are above the limit. keep last message user and assistant message and truncate last assistant message.
        if token_count > self.token_limit:
            return self._get_truncated_messages(messages=chat_history)

        # catch one message longer than token limit
        # if token_count > self.token_limit or (message_count <= 0 and not self.system_prompt):
        #     return []

        if self.system_prompt:
            return [Message(role="system", content=self.system_prompt)] + chat_history[-message_count:]
        
        return chat_history[-message_count:]
