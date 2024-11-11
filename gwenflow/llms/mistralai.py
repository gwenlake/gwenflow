from typing import Optional, Union, Mapping, Any, List, Dict, Iterator, AsyncIterator
import os
import logging
from mistralai.client import MistralClient

from gwenflow.llms.base import ChatBase
from gwenflow.types import ChatCompletion, ChatCompletionChunk


logger = logging.getLogger(__name__)


class ChatMistralAI(ChatBase):
 
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str,
        temperature: Optional[float] = 0.0,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,            
    ):
        _api_key = api_key or os.environ.get("MISTRAL_API_KEY")

        self.client = MistralClient(api_key=_api_key)

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
 
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
        }

        if response_format:
            params["response_format"] = response_format

        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice
    
        response = self.client.chat.completions.create(**params)
        return ChatCompletion(**response.dict())

    def stream(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Any] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",            
    ) -> Iterator[Mapping[str, Any]]:

        params = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "stream": True,
        }

        if response_format:
            params["response_format"] = response_format

        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice

        content = ""
        response = self.client.chat.completions.create(**params)
        for chunk in response:
            if chunk.choices[0].delta.content:
                content += chunk.choices[0].delta.content
            yield ChatCompletionChunk(**chunk.dict())
