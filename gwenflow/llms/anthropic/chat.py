import json
import os
from collections.abc import AsyncIterator
from typing import Any, Dict, Iterator, List, Optional, Type, Union

import anthropic
from pydantic import BaseModel

from gwenflow.llms.base import ChatBase
from gwenflow.telemetry import tracer
from gwenflow.types import Function, ItemHelpers, Message, ModelResponse, ToolCall, Usage
from gwenflow.utils import extract_json_str, make_pydantic_schema_strict_json


class ChatAnthropic(ChatBase):
    model: str = "claude-sonnet-4-6"

    # model parameters
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stop_sequences: Optional[List[str]] = None
    max_tokens: Optional[int] = 4096
    response_format: Optional[Union[Dict[str, Any], Type[BaseModel]]] = None

    # clients
    client: Optional[anthropic.Anthropic] = None
    async_client: Optional[anthropic.AsyncAnthropic] = None

    # client parameters
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: Optional[Union[float, int]] = None
    max_retries: Optional[int] = None

    # client parameters
    system_prompt: Optional[str] = None

    def _get_client_params(self) -> Dict[str, Any]:
        api_key = self.api_key
        if api_key is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key is None:
            raise ValueError(
                "The api_key client option must be set either by passing api_key to the client or by setting the ANTHROPIC_API_KEY environment variable"
            )

        client_params = {
            "api_key": api_key,
            "base_url": self.base_url,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }

        return {k: v for k, v in client_params.items() if v is not None}

    @property
    def _model_params(self) -> Dict[str, Any]:
        model_params = {
            "max_tokens": self.max_tokens or 4096,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "stop_sequences": self.stop_sequences,
        }

        if self.tools and self.tool_type == "fncall":
            model_params["tools"] = [self._tool_to_anthropic(tool) for tool in self.tools]
            if isinstance(self.tool_choice, dict):
                model_params["tool_choice"] = self.tool_choice
            else:
                model_params["tool_choice"] = {"type": "auto"}

        if self.response_format:
            if isinstance(self.response_format, type) and issubclass(self.response_format, BaseModel):
                raw_schema = self.response_format.model_json_schema()
                strict_schema = make_pydantic_schema_strict_json(raw_schema)

                model_params["output_config"] = {"format": {"type": "json_schema", "schema": strict_schema}}
            else:
                model_params["output_config"] = self.response_format
        return {k: v for k, v in model_params.items() if v is not None}

    def _tool_to_anthropic(self, tool) -> Dict[str, Any]:
        return {
            "name": tool.name,
            "description": tool.description or "",
            "input_schema": tool.params_json_schema,
        }

    def get_client(self) -> anthropic.Anthropic:
        if self.client:
            return self.client
        self.client = anthropic.Anthropic(**self._get_client_params())
        return self.client

    def get_async_client(self) -> anthropic.AsyncAnthropic:
        if self.async_client:
            return self.async_client
        self.async_client = anthropic.AsyncAnthropic(**self._get_client_params())
        return self.async_client

    def _format_response(self, response: str, response_format: Any = None) -> Any:
        if response is None:
            return None
        json_dict = isinstance(response_format, dict) and response_format.get("type") == "json_object"
        pydantic = isinstance(response_format, type) and issubclass(response_format, BaseModel)
        if json_dict or pydantic:
            try:
                return json.loads(extract_json_str(response))
            except (json.JSONDecodeError, TypeError, ValueError):
                return response
        return response

    def _format_messages(self, messages: List[Message]) -> tuple:
        self.system_prompt = None
        formatted = []

        for message in messages:
            if message.role == "system":
                self.system_prompt = message.content
                continue

            if message.role == "tool":
                formatted.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": message.tool_call_id,
                                "content": message.content or "",
                            }
                        ],
                    }
                )
            elif message.role == "assistant" and message.tool_calls:
                content = []
                if message.content:
                    content.append({"type": "text", "text": message.content})
                for tc in message.tool_calls:
                    if isinstance(tc, dict):
                        tc_id = tc["id"]
                        tc_name = tc["function"]["name"]
                        tc_args = tc["function"]["arguments"]
                    else:
                        tc_id = tc.id
                        tc_name = tc.function.name
                        tc_args = tc.function.arguments
                    try:
                        input_data = json.loads(tc_args)
                    except (json.JSONDecodeError, TypeError):
                        input_data = {}
                    content.append({"type": "tool_use", "id": tc_id, "name": tc_name, "input": input_data})
                formatted.append({"role": "assistant", "content": content})
            else:
                if isinstance(message.content, list):
                    content = [
                        item if isinstance(item, dict) else {"type": "text", "text": str(item)}
                        for item in message.content
                    ]
                    formatted.append({"role": message.role, "content": content})
                else:
                    formatted.append({"role": message.role, "content": message.content or ""})

        return formatted

    def _get_usage(self, usage) -> Optional[Usage]:
        if usage is None:
            return None
        input_tokens = getattr(usage, "input_tokens", 0) or 0
        output_tokens = getattr(usage, "output_tokens", 0) or 0
        return Usage(
            requests=1,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )

    def _parse_response(self, response) -> ModelResponse:
        model_response = ModelResponse(
            role="assistant",
            finish_reason="stop",
            usage=self._get_usage(response.usage),
        )

        for block in response.content:
            if block.type == "text":
                model_response.content = block.text
            elif block.type == "thinking":
                model_response.reasoning_content = block.thinking
            elif block.type == "tool_use":
                model_response.tool_calls.append(
                    ToolCall(
                        id=block.id,
                        type="function",
                        function=Function(name=block.name, arguments=json.dumps(block.input)),
                    )
                )

        if response.stop_reason == "tool_use":
            model_response.finish_reason = "tool_calls"

        if self.response_format:
            model_response.parsed = self._format_response(model_response.content, response_format=self.response_format)

        return model_response

    def _build_request(self, messages: List[Message]) -> Dict[str, Any]:
        formatted_messages = self._format_messages(messages)
        payload = {"model": self.model, "messages": formatted_messages, **self._model_params}
        if self.system_prompt:
            payload["system"] = self.system_prompt
        return payload

    @tracer.llm(name="LLM Invoke")
    def invoke(self, input: Union[str, List[Message], List[Dict[str, str]]]) -> ModelResponse:
        try:
            messages_for_model = ItemHelpers.input_to_message_list(input)
            response = self.get_client().messages.create(**self._build_request(messages_for_model))
        except Exception as e:
            raise RuntimeError(f"Error in calling anthropic API: {e}") from e
        return self._parse_response(response)

    @tracer.llm(name="LLM Async Invoke")
    async def ainvoke(self, input: Union[str, List[Message], List[Dict[str, str]]]) -> ModelResponse:
        try:
            messages_for_model = ItemHelpers.input_to_message_list(input)
            response = await self.get_async_client().messages.create(**self._build_request(messages_for_model))
        except Exception as e:
            raise RuntimeError(f"Error in calling anthropic API: {e}") from e
        return self._parse_response(response)

    @tracer.llm(name="LLM Stream")
    def stream(self, input: Union[str, List[Message], List[Dict[str, str]]]) -> Iterator[ModelResponse]:
        try:
            messages_for_model = ItemHelpers.input_to_message_list(input)
            request = self._build_request(messages_for_model)
            _full_tool_calls: Dict[int, ToolCall] = {}
            _input_tokens = 0

            with self.get_client().messages.stream(**request) as stream:
                for event in stream:
                    response = ModelResponse(role="assistant")

                    if event.type == "message_start":
                        _input_tokens = event.message.usage.input_tokens

                    elif event.type == "content_block_start":
                        if event.content_block.type == "tool_use":
                            _full_tool_calls[event.index] = ToolCall(
                                id=event.content_block.id,
                                type="function",
                                function=Function(name=event.content_block.name, arguments=""),
                            )

                    elif event.type == "content_block_delta":
                        if event.delta.type == "text_delta":
                            response.content = event.delta.text
                            yield response
                        elif event.delta.type == "input_json_delta":
                            if event.index in _full_tool_calls:
                                _full_tool_calls[event.index].function.arguments += event.delta.partial_json
                        elif event.delta.type == "thinking_delta":
                            response.reasoning_content = event.delta.thinking
                            yield response

                    elif event.type == "message_delta":
                        if event.delta.stop_reason == "tool_use" and _full_tool_calls:
                            response.tool_calls = list(_full_tool_calls.values())
                            response.finish_reason = "tool_calls"
                            yield response
                        output_tokens = getattr(event.usage, "output_tokens", 0) or 0
                        if output_tokens:
                            response.usage = Usage(
                                requests=1,
                                input_tokens=_input_tokens,
                                output_tokens=output_tokens,
                                total_tokens=_input_tokens + output_tokens,
                            )
                            yield response

        except Exception as e:
            raise RuntimeError(f"Error in calling anthropic API: {e}") from e

    @tracer.llm(name="LLM Async Astream")
    async def astream(self, input: Union[str, List[Message], List[Dict[str, str]]]) -> AsyncIterator[ModelResponse]:
        try:
            messages_for_model = ItemHelpers.input_to_message_list(input)
            request = self._build_request(messages_for_model)
            _full_tool_calls: Dict[int, ToolCall] = {}
            _input_tokens = 0

            async with self.get_async_client().messages.stream(**request) as stream:
                async for event in stream:
                    response = ModelResponse(role="assistant")

                    if event.type == "message_start":
                        _input_tokens = event.message.usage.input_tokens

                    elif event.type == "content_block_start":
                        if event.content_block.type == "tool_use":
                            _full_tool_calls[event.index] = ToolCall(
                                id=event.content_block.id,
                                type="function",
                                function=Function(name=event.content_block.name, arguments=""),
                            )

                    elif event.type == "content_block_delta":
                        if event.delta.type == "text_delta":
                            response.content = event.delta.text
                            yield response
                        elif event.delta.type == "input_json_delta":
                            if event.index in _full_tool_calls:
                                _full_tool_calls[event.index].function.arguments += event.delta.partial_json
                        elif event.delta.type == "thinking_delta":
                            response.reasoning_content = event.delta.thinking
                            yield response

                    elif event.type == "message_delta":
                        if event.delta.stop_reason == "tool_use" and _full_tool_calls:
                            response.tool_calls = list(_full_tool_calls.values())
                            response.finish_reason = "tool_calls"
                            yield response
                        output_tokens = getattr(event.usage, "output_tokens", 0) or 0
                        if output_tokens:
                            response.usage = Usage(
                                requests=1,
                                input_tokens=_input_tokens,
                                output_tokens=output_tokens,
                                total_tokens=_input_tokens + output_tokens,
                            )
                            yield response

        except Exception as e:
            raise RuntimeError(f"Error in calling anthropic API: {e}") from e
