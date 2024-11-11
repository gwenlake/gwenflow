from typing import Optional, Union, Mapping, Any, List, Dict, Iterator, AsyncIterator
import os
import logging
import anthropic

from gwenflow.llms.base import ChatBase
from gwenflow.types import ChatCompletion, ChatCompletionChunk


logger = logging.getLogger(__name__)


class ChatAnthropic(ChatBase):
 
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str,
        temperature: Optional[float] = 0.0,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
    ):
        _api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

        self.client = anthropic.Anthropic(api_key=_api_key)

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Any] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
    ) -> ChatCompletion:
        
        messages = self._get_messages(messages)
        system   = self._get_system(messages)

        params = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
        }

        if system:
            params["system"] = system["content"]

        response = self.client.messages.create(**params)
        return ChatCompletion(**response.dict())


    def stream(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Any] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",            
    ) -> Iterator[Mapping[str, Any]]:
        
        messages = self._get_messages(messages)
        system   = self._get_system(messages)

        params = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "stream": True,
        }

        if system:
            params["system"] = system["content"]

        if response_format:
            params["response_format"] = response_format

        response = self.client.messages.create(**params)

        content = ""
        for chunk in response:
            if chunk.type != "content_block_stop":
                if chunk.delta.text:
                    content += chunk.delta.text
            yield ChatCompletionChunk(**chunk.dict())
