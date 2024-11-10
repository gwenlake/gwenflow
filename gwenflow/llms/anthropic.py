from typing import Optional, Union
import os
import logging
import anthropic

from gwenflow.base.types import Usage, ChatMessage
from gwenflow.utils.tokens import num_tokens_from_string, num_tokens_from_messages
from gwenflow.llms.base import ChatBase


logger = logging.getLogger(__name__)


class ChatAnthropic(ChatBase):
 
    def __init__(self, *, api_key: Optional[str] = None, model: str, temperature=0.0):
        _api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.temperature = temperature
        self.model = model
        self.client = anthropic.Anthropic(api_key=_api_key)

    def chat(self, messages: Union[list[ChatMessage], ChatMessage, str]):
        try:
            messages = self._get_messages(messages)
            system   = self._get_system(messages)
            if system:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    system=system,
                    messages=messages,
                    temperature=self.temperature
                )
            else:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    messages=messages,
                    temperature=self.temperature
                )
        except Exception as e:
            logger.error(e)
            return None
        return response.dict()

    def stream(self, messages: Union[list[ChatMessage], ChatMessage, str]):
        try:
            messages = self._get_messages(messages)
            system   = self._get_system(messages)
            if system:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    system=system,
                    messages=messages,
                    temperature=self.temperature,
                    stream=True
                )
            else:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    messages=messages,
                    temperature=self.temperature,
                    stream=True
                )
        except Exception as e:
            logger.error(e)
            yield ""
    
        content = ""
        for chunk in response:
            chunk = chunk.dict()
            if chunk["type"] != "content_block_stop":
                if "delta" in chunk:
                    if "text" in chunk["delta"]:
                        content += chunk["delta"]["text"]
            yield chunk
