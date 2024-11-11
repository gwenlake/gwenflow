import os
import logging
import openai
from typing import Optional, Union, Mapping, Any
from typing import Iterator, AsyncIterator

from gwenflow.base.types import ChatMessage
from gwenflow.llms.base import ChatBase


logger = logging.getLogger(__name__)


class OpenAI(ChatBase):
 
    def __init__(self, *, api_key: Optional[str] = None, model: str, temperature=0.0):
        _api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if os.environ.get('OPENAI_ORG_ID'):
            openai.organization = os.environ.get('OPENAI_ORG_ID')
        self.temperature = temperature
        self.model = model
        self.client = openai.OpenAI(api_key=_api_key)

    def chat(self, messages: Union[list[ChatMessage], ChatMessage, str]):
        try:
            messages = self._get_messages(messages)
            response = self.client.chat.completions.create(model=self.model, messages=messages, temperature=self.temperature)
        except Exception as e:
            logger.error(e)
            return None
        if not response.choices[0].message.content:
            return None
        return response.dict()

    def stream(self, messages: Union[list[ChatMessage], ChatMessage, str]) -> Iterator[Mapping[str, Any]]:
        try:
            messages = self._get_messages(messages)
            response = self.client.chat.completions.create(model=self.model, messages=messages, temperature=self.temperature, stream=True) #,  stream_options={"include_usage": True})
        except Exception as e:
            logger.error(e)
            yield ""
    
        content = ""
        for chunk in response:
            if chunk.choices[0].delta.content:
                content += chunk.choices[0].delta.content
            yield chunk.dict()
 