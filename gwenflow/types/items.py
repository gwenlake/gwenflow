import copy
from typing import Dict, List, Union

from .message import Message


class ItemHelpers:
    @classmethod
    def input_to_message_list(
        cls,
        input: Union[str, List[Message], List[Dict[str, str]]],
    ) -> List[Message]:
        """Converts a string or list of messages into a list of messages."""
        if isinstance(input, str):
            return [Message(role="user", content=input)]
        messages = copy.deepcopy(input)
        for i, message in enumerate(messages):
            if isinstance(message, dict):
                messages[i] = Message(**message)
        return messages
