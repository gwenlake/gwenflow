from typing import Optional, Union, Mapping, Any, List, Dict, Iterator, AsyncIterator
import os
import logging
import openai

from gwenflow.llms.base import ChatBase
from gwenflow.types import ChatCompletion, ChatCompletionChunk


logger = logging.getLogger(__name__)


class ChatOpenAI(ChatBase):
 
    def __init__(
        self,
        *,
        model: str,
        timeout: Optional[Union[float, int]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        n: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        max_completion_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        logit_bias: Optional[Dict[int, float]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        base_url: Optional[str] = None,
        api_version: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            model = model,
            timeout = timeout,
            temperature = temperature,
            top_p = top_p,
            n = n,
            stop = stop,
            max_completion_tokens = max_completion_tokens,
            max_tokens = max_tokens,
            presence_penalty = presence_penalty,
            frequency_penalty = frequency_penalty,
            logit_bias = logit_bias,
            response_format = response_format,
            seed = seed,
            logprobs = logprobs,
            top_logprobs = top_logprobs,
            **kwargs,
        )

        _api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if os.environ.get('OPENAI_ORG_ID'):
            openai.organization = os.environ.get('OPENAI_ORG_ID')

        self.client = openai.OpenAI(api_key=_api_key)
    
    def invoke(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Any] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
    ) -> ChatCompletion:
 
        params = self.config
        params["messages"] = messages

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

        params = self.config
        params["messages"] = messages
        params["stream"] = True

        if response_format:
            params["response_format"] = response_format

        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice

        response = self.client.chat.completions.create(**params)

        content = ""
        for chunk in response:
            if chunk.choices[0].delta.content:
                content += chunk.choices[0].delta.content
            yield ChatCompletionChunk(**chunk.dict())
 