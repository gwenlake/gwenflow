from abc import ABC, abstractmethod
from typing import Optional, Union

from gwenflow.base.types import ChatMessage


class ChatBase(ABC):
 
    def _get_system(self, input: Union[list[ChatMessage], ChatMessage, str]):
        if isinstance(input, list):
            for message in input:
                if message.role == "system":
                    return message.content
        elif isinstance(input, ChatMessage):
            if input.role == "system":
                return input.content
        return None

    def _get_messages(self, input: Union[list[ChatMessage], ChatMessage, str]):
        if isinstance(input, list):
            messages = []
            for message in input:
                if isinstance(message, dict):
                    message = ChatMessage(**message)
                if message.role != "system":
                    messages.append(message)
            return messages
        elif isinstance(input, ChatMessage):
            if isinstance(message, dict):
                input = ChatMessage(**input)
            if input.role != "system":
                return [input]
        elif isinstance(input, str):
            return [ChatMessage(role="user", content=input)]
        return None

    @abstractmethod
    def chat(self, messages: Union[list[ChatMessage], ChatMessage, str]):
        pass

    @abstractmethod
    def stream(self, messages: Union[list[ChatMessage], ChatMessage, str]):
        pass
