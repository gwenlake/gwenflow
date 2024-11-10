from typing import Optional, Union
import os
import logging
from mistralai.client import MistralClient
# from mistralai.models.chat_completion import ChatMessage

from gwenflow.base.types import ChatMessage
from gwenflow.llms.base import ChatBase


logger = logging.getLogger(__name__)


class MistralAI(ChatBase):
 
    def __init__(self, *, api_key: Optional[str] = None, model: str, temperature=0.0):
        _api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        self.temperature = temperature
        self.model = model
        self.client = MistralClient(api_key=_api_key)

    def chat(self, messages: Union[list[ChatMessage], ChatMessage, str]):
        try:
            messages = self._get_messages(messages)
            response = self.client.chat(model=self.model, messages=messages, temperature=self.temperature)
        except Exception as e:
            logger.error(e)
            return None
        return response.dict()

    def stream(self, messages: Union[list[ChatMessage], ChatMessage, str]):
        try:
            messages = self._get_messages(messages)
            response = self.client.chat_stream(model=self.model, messages=messages, temperature=self.temperature)
        except Exception as e:
            logger.error(e)
            yield ""
    
        content = ""
        for chunk in response:
            if not chunk.choices[0].finish_reason:
                if chunk.choices[0].delta.content:
                    content += chunk.choices[0].delta.content
            yield chunk.dict()
