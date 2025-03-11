import os
import logging
import json
import dirtyjson

from collections import defaultdict
from typing import Optional, Union, Any, List, Dict
from openai import OpenAI, AsyncOpenAI

from gwenflow.types import ChatCompletion, ChatCompletionChunk
from gwenflow.llms.base import ChatBase
from gwenflow.utils.chunks import merge_chunk
from gwenflow.utils import extract_json_str


logger = logging.getLogger(__name__)

MAX_LOOPS = 10

class ChatOpenAI(ChatBase):
 
    model: str = "gpt-4o-mini"

    # model parameters
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    max_completion_tokens: Optional[int] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[Dict[int, float]] = None
    response_format: Optional[Dict[str, Any]] = None
    seed: Optional[int] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None

    # clients
    client: Optional[OpenAI] = None
    async_client: Optional[AsyncOpenAI] = None

    # client parameters
    api_key: Optional[str] = None
    organization: Optional[str] = None
    base_url: Optional[str] = None
    timeout: Optional[Union[float, int]] = None
    max_retries: Optional[int] = None

    def _get_client_params(self) -> Dict[str, Any]:

        api_key = self.api_key
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError(
                "The api_key client option must be set either by passing api_key to the client or by setting the OPENAI_API_KEY environment variable"
            )

        organization = self.organization
        if organization is None:
            organization = os.environ.get('OPENAI_ORG_ID')

        client_params = {
            "api_key": api_key,
            "organization": organization,
            "base_url": self.base_url,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }

        client_params = {k: v for k, v in client_params.items() if v is not None}

        return client_params

    # @property
    def _get_model_params(self, tools: list = None, tool_choice: str = None, response_format: str = None) -> Dict[str, Any]:

        model_params = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "n": self.n,
            "stop": self.stop,
            "max_tokens": self.max_tokens or self.max_completion_tokens,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "logit_bias": self.logit_bias,
            "response_format": self.response_format,
            "seed": self.seed,
            "logprobs": self.logprobs,
            "top_logprobs": self.top_logprobs,
        }

        if tools:
            model_params["tools"] = tools
            model_params["tool_choice"] = tool_choice or "auto"
        elif self.tools:
            tools = [tool.openai_schema for tool in self.tools]
            model_params["tools"] = tools or None
            model_params["tool_choice"] = self.tool_choice or "auto"
        
        if response_format:
            model_params["response_format"] = response_format

        model_params = {k: v for k, v in model_params.items() if v is not None}

        return model_params
    
    def get_client(self) -> OpenAI:

        if self.client:
            return self.client
        
        client_params = self._get_client_params()

        self.client = OpenAI(**client_params)
        return self.client

    def get_async_client(self) -> AsyncOpenAI:

        if self.client:
            return self.client
        
        client_params = self._get_client_params()

        self.async_client = AsyncOpenAI(**client_params)
        return self.async_client

    def _parse_response(self, response: str, response_format: dict = None) -> str:
        """Process the response."""

        if response_format.get("type") == "json_object":
            try:
                json_str = extract_json_str(response)
                # text_response = dirtyjson.loads(json_str)
                text_response = json.loads(json_str)
                return text_response
            except:
                pass

        return response

    def invoke(
        self,
        messages: Union[str, List[Dict[str, str]]],
        tools: list = None,
        tool_choice: str = None,
        response_format: str = None,
    ) -> ChatCompletion:

        if isinstance(messages, str):
            messages = [{ "role": "user", "content": messages }]

        response = self.get_client().chat.completions.create(
            model=self.model,
            messages=messages,
            **self._get_model_params(tools=tools, tool_choice=tool_choice, response_format=response_format),
        )

        response = ChatCompletion(**response.model_dump())

        if self.response_format or response_format:
            _response_format = response_format or self.response_format
            response.choices[0].message.content = self._parse_response(response.choices[0].message.content, response_format=_response_format)

        return response

    async def ainvoke(
        self,
        messages: Union[str, List[Dict[str, str]]],
        tools: list = None,
        tool_choice: str = None,
        response_format: str = None,
    ) -> ChatCompletion:

        if isinstance(messages, str):
            messages = [{ "role": "user", "content": messages }]

        response = await self.get_async_client().chat.completions.create(
            model=self.model,
            messages=messages,
            **self._get_model_params(tools=tools, tool_choice=tool_choice, response_format=response_format),
        )

        response = ChatCompletion(**response.model_dump())

        if self.response_format or response_format:
            _response_format = response_format or self.response_format
            response.choices[0].message.content = self._parse_response(response.choices[0].message.content, response_format=_response_format)

        return response

    def stream(
        self,
        messages: Union[str, List[Dict[str, str]]],
        tools: list = None,
        tool_choice: str = None,
        response_format: str = None,
    ):

        if isinstance(messages, str):
            messages = [{ "role": "user", "content": messages }]

        response = self.get_client().chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
            **self._get_model_params(tools=tools, tool_choice=tool_choice, response_format=response_format),
        )

        for chunk in response:
            chunk = ChatCompletionChunk(**chunk.model_dump())
            yield chunk
    
    async def astream(
        self,
        messages: Union[str, List[Dict[str, str]]],
        tools: list = None,
        tool_choice: str = None,
        response_format: str = None,
    ):

        if isinstance(messages, str):
            messages = [{ "role": "user", "content": messages }]

        response = await self.get_async_client().chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
            **self._get_model_params(tools=tools, tool_choice=tool_choice, response_format=response_format),
        )

        async for chunk in response:
            chunk = ChatCompletionChunk(**chunk.model_dump())
            yield chunk
