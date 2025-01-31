from typing import Optional, Union, Mapping, Any, List, Dict, Iterator, AsyncIterator
import os
import logging
import json
import openai
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

from gwenflow.llms.base import ChatBase


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

    def _parse_response(self, response, tools):
        """
        Process the response based on whether tools are used or not.

        Args:
            response: The raw response from API.
            tools: The list of tools provided in the request.

        Returns:
            str or dict: The processed response.
        """
        if tools:
            processed_response = {
                "content": response.choices[0].message.content,
                "tool_calls": [],
            }

            if response.choices[0].message.tool_calls:
                for tool_call in response.choices[0].message.tool_calls:
                    processed_response["tool_calls"].append(
                        {
                            "name": tool_call.function.name,
                            "arguments": json.loads(tool_call.function.arguments),
                        }
                    )

            return processed_response
        
        if isinstance(response, ChatCompletionChunk):
            if response.choices[0].delta.content:
                return response.choices[0].delta.content
            return ""
        
        return response.choices[0].message.content
        

    def invoke(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Any] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        parallel_tool_calls: Optional[bool] = None,
        parse_response: bool = True,
    ):
 
        params = self.config
        params["messages"] = messages

        if response_format:
            params["response_format"] = response_format
        else:
            params["response_format"] = None

        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice
            if parallel_tool_calls:
                params["parallel_tool_calls"] = parallel_tool_calls
        else:
            params["tools"] = None
            params["tool_choice"] = None

        response = self.client.chat.completions.create(**params)
        if parse_response:
            response = self._parse_response(response, tools)
        return response

    def stream(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Any] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        parse_response: bool = True,
    ):

        params = self.config
        params["messages"] = messages
        params["stream"] = True

        if response_format:
            params["response_format"] = response_format
        else:
            params["response_format"] = None

        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice
        else:
            params["tools"] = None
            params["tool_choice"] = None

        response = self.client.chat.completions.create(**params)

        content = ""
        for chunk in response:
            if len(chunk.choices) > 0:
                if chunk.choices[0].finish_reason == "stop":
                    break
                if chunk.choices[0].delta.content:
                    content += chunk.choices[0].delta.content
                if parse_response:
                    chunk = self._parse_response(chunk, tools)
                yield chunk
 