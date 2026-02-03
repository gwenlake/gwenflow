import dirtyjson
import json
import os
from collections.abc import AsyncIterator
from typing import Optional, Union, Any, List, Dict, Iterator
from pydantic import BaseModel

from gwenflow.logger import logger
from gwenflow.llms import ChatBase
from gwenflow.types import Message, ItemHelpers, ModelResponse, Usage, ToolCall
from gwenflow.utils import extract_json_str

from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import ChoiceDelta


    
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

    @property
    def _model_params(self) -> Dict[str, Any]:

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

        if self.tools and self.tool_type == "fncall":
            model_params["tools"] = [tool.to_openai() for tool in self.tools]
            model_params["tool_choice"] = self.tool_choice or "auto"
        
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

    def _format_response(self, response: str, response_format: dict = None) -> str:
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

    def _format_message(self, message: Message) -> Dict[str, Any]:
        """Format a message into the format expected by OpenAI."""
        return message.to_openai()
    
    def _get_openai_usage(self, completion: Union[ChatCompletion, ChatCompletionChunk]) -> Usage:
        if not completion.usage:
            return None
        return Usage(
            requests=1,
            input_tokens=completion.usage.prompt_tokens,
            output_tokens=completion.usage.completion_tokens,
            total_tokens=completion.usage.total_tokens,
        )
    
    def _parse_response(self, completion: ChatCompletion) -> ModelResponse:
        model_response = ModelResponse(
            role=completion.choices[0].message.role,
            content=completion.choices[0].message.content,
            finish_reason="stop",
            usage=self._get_openai_usage(completion),
        )

        if hasattr(completion.choices[0].message, 'reasoning_content'):
            model_response.reasoning_content = completion.choices[0].message.reasoning_content

        tool_calls = completion.choices[0].message.tool_calls
        if tool_calls is not None and len(tool_calls) > 0:
            try:
                model_response.tool_calls = [ToolCall(**t.model_dump()) for t in tool_calls]
            except Exception as e:
                logger.warning(f"Error processing tool calls: {e}")

        if self.response_format:
            model_response.parsed = self._format_response(completion.choices[0].message.content, response_format=self.response_format)
        
        return model_response

    def invoke(self, input: Union[str, List[Message], List[Dict[str, str]]]) -> ModelResponse:
        try:
            messages_for_model = ItemHelpers.input_to_message_list(input)
            response: ChatCompletion = self.get_client().chat.completions.create(
                model=self.model,
                messages=[self._format_message(m) for m in messages_for_model],
                **self._model_params,
            )
        except Exception as e:
            raise RuntimeError(f"Error in calling openai API: {e}")

        return self._parse_response(response)

    async def ainvoke(self, input: Union[str, List[Message], List[Dict[str, str]]]) -> ModelResponse:
        try:
            messages_for_model = ItemHelpers.input_to_message_list(input)
            response: ChatCompletion = await self.get_async_client().chat.completions.create(
                model=self.model,
                messages=[self._format_message(m) for m in messages_for_model],
                **self._model_params,
            )
        except Exception as e:
            raise RuntimeError(f"Error in calling openai API: {e}")

        return self._parse_response(response)

    def stream(self, input: Union[str, List[Message], List[Dict[str, str]]]) -> Iterator[ModelResponse]:
        try:
            messages_for_model = ItemHelpers.input_to_message_list(input)
            completion = self.get_client().chat.completions.create(
                model=self.model,
                messages=[self._format_message(m) for m in messages_for_model],
                stream=True,
                stream_options={"include_usage": True},
                **self._model_params,
            )

            _full_tool_calls = []

            for chunk in completion:
                
                response = ModelResponse(role="assistant")

                if chunk.choices:

                    delta: ChoiceDelta = chunk.choices[0].delta

                    if hasattr(delta, 'content') and delta.content:
                        response.content = delta.content

                    if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                        response.reasoning_content = delta.reasoning_content

                    if hasattr(delta, 'tool_calls') and delta.tool_calls:
                        for tool_call in delta.tool_calls or []:
                            if _full_tool_calls and (not tool_call.id or tool_call.id == _full_tool_calls[-1].id):
                                if tool_call.function.name:
                                    _full_tool_calls[-1].function.name += tool_call.function.name
                                if tool_call.function.arguments:
                                    _full_tool_calls[-1].function.arguments += tool_call.function.arguments
                            else:
                                _full_tool_calls.append(
                                    ToolCall(**tool_call.model_dump())
                                )

                    if _full_tool_calls:
                        response.tool_calls = _full_tool_calls

                    if response.content or response.reasoning_content:
                        yield response
                    elif len(response.tool_calls)>0 and chunk.choices[0].finish_reason == "tool_calls":
                        yield response

                else:
                    if hasattr(response, 'usage'):
                        response.usage = self._get_openai_usage(chunk)
                        yield response

        except Exception as e:
            raise RuntimeError(f"Error in calling openai API: {e}")

    async def astream(self, input: Union[str, List[Message], List[Dict[str, str]]]) -> AsyncIterator[ModelResponse]:
        try:
            messages_for_model = ItemHelpers.input_to_message_list(input)
            completion = await self.get_async_client().chat.completions.create(
                model=self.model,
                messages=[self._format_message(m) for m in messages_for_model],
                stream=True,
                stream_options={"include_usage": True},
                **self._model_params,
            )

            _full_tool_calls = []

            async for chunk in completion:

                response = ModelResponse(role="assistant")

                if chunk.choices:

                    delta: ChoiceDelta = chunk.choices[0].delta

                    if hasattr(delta, 'content') and delta.content:
                        response.content = delta.content

                    if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                        response.reasoning_content = delta.reasoning_content

                    if hasattr(delta, 'tool_calls') and delta.tool_calls:
                        for tool_call in delta.tool_calls or []:
                            if _full_tool_calls and (not tool_call.id or tool_call.id == _full_tool_calls[-1].id):
                                if tool_call.function.name:
                                    _full_tool_calls[-1].function.name += tool_call.function.name
                                if tool_call.function.arguments:
                                    _full_tool_calls[-1].function.arguments += tool_call.function.arguments
                            else:
                                _full_tool_calls.append(
                                    ToolCall(**tool_call.model_dump())
                                )

                    if _full_tool_calls:
                        response.tool_calls = _full_tool_calls

                    if response.content or response.reasoning_content:
                        yield response
                    elif len(response.tool_calls)>0 and chunk.choices[0].finish_reason == "tool_calls":
                        yield response

                else:
                    if hasattr(response, 'usage'):
                        response.usage = self._get_openai_usage(chunk)
                        yield response

        except Exception as e:
            raise RuntimeError(f"Error in calling openai API: {e}")
