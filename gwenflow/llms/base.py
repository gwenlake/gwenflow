from abc import ABC, abstractmethod
from typing import Optional, Union, Mapping, Any, List, Dict, Iterator, AsyncIterator


class ChatBase(ABC):
 
    def _get_system(self, messages: list[dict]):
        for message in messages:
            if message["role"] == "system":
                return message
        return None

    def _get_messages(self, messages: list[dict]):
        filtered_messages = []
        for message in messages:
            if message["role"] != "system":
                filtered_messages.append(message)
        return filtered_messages

    @abstractmethod
    def generate(self, messages):
        """
        Generate a response based on the given messages.

        Args:
            messages (list): List of message dicts containing 'role' and 'content'.
            response_format (str or object, optional): Format of the response. Defaults to "text".
            tools (list, optional): List of tools that the model can call. Defaults to None.
            tool_choice (str, optional): Tool choice method. Defaults to "auto".

        Returns:
            str: The generated response.
        """
        pass

    @abstractmethod
    def stream(self, messages):
        """
        Generate a streamed response based on the given messages.

        Args:
            messages (list): List of message dicts containing 'role' and 'content'.
            response_format (str or object, optional): Format of the response. Defaults to "text".
            tools (list, optional): List of tools that the model can call. Defaults to None.
            tool_choice (str, optional): Tool choice method. Defaults to "auto".

        Returns:
            str: The generated response.
        """
        pass
