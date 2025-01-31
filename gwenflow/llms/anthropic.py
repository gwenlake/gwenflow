from typing import Optional, Union, Any, List, Dict
import os
import logging
import anthropic

from gwenflow.llms.base import ChatBase


logger = logging.getLogger(__name__)


class ChatAnthropic(ChatBase):
 
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

        _api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

        self.client = anthropic.Anthropic(api_key=_api_key)
    
    def invoke(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Any] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
    ):
        
        messages = self._get_messages(messages)
        system   = self._get_system(messages)

        params = self.config
        params["messages"] = messages

        if system:
            params["system"] = system["content"]

        # Remove None values to avoid passing unnecessary parameters
        params = {k: v for k, v in params.items() if v is not None}

        response = self.client.messages.create(**params)
        return ChatCompletion(**response.dict())


    def stream(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Any] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",            
    ):
        
        messages = self._get_messages(messages)
        system   = self._get_system(messages)

        params = self.config
        params["messages"] = messages
        params["stream"] = True

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
