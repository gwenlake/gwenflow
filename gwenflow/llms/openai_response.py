import json
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Literal, Optional, Type, Union

from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel

from gwenflow.llms.base import ChatBase
from gwenflow.telemetry import tracer
from gwenflow.tools import Tool
from gwenflow.types import Message, ModelResponse, RequestUsage, TextContent, ThinkingContent, ToolCall
from gwenflow.utils import extract_json_str, make_pydantic_schema_strict_json


@dataclass(kw_only=True)
class ResponseOpenAI(ChatBase):
    model: str = "gpt-5-mini"

    # model parameters
    background: Optional[bool] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_output_tokens: Optional[int] = None
    max_tool_calls: Optional[int] = None
    prompt_cache_key: Optional[str] = None
    prompt_cache_retention: Optional[str] = None
    top_logprobs: Optional[int] = None
    reasoning_effort: Optional[Literal["low", "medium", "high"]] = None
    reasoning_summary: Optional[Literal["auto", "concise", "detailed"]] = None
    response_format: Optional[Union[Dict[str, Any], Type[BaseModel]]] = None
    show_reasoning: bool = False

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
        api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError(
                "The api_key client option must be set either by passing api_key to the client or by setting the OPENAI_API_KEY environment variable"
            )

        organization = self.organization or os.environ.get("OPENAI_ORG_ID")

        client_params = {
            "api_key": api_key,
            "organization": organization,
            "base_url": self.base_url,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }
        return {k: v for k, v in client_params.items() if v is not None}

    @property
    def _model_params(self) -> Dict[str, Any]:
        model_params: Dict[str, Any] = {
            "background": self.background,
            "max_output_tokens": self.max_output_tokens,
            "max_tool_calls": self.max_tool_calls,
            "prompt_cache_key": self.prompt_cache_key,
            "prompt_cache_retention": self.prompt_cache_retention,
            "temperature": self.temperature,
            "top_logprobs": self.top_logprobs,
            "top_p": self.top_p,
        }

        if self.get_reasoning_model() and (self.reasoning_effort or self.reasoning_summary):
            reasoning: Dict[str, Any] = {}
            if self.reasoning_effort:
                reasoning["effort"] = self.reasoning_effort
            if self.reasoning_summary:
                reasoning["summary"] = self.reasoning_summary
            model_params["reasoning"] = reasoning

        if self.tools and self.tool_type == "fncall":
            model_params["tools"] = [self._tool_to_responses(tool) for tool in self.tools]
            model_params["tool_choice"] = self.tool_choice or "auto"

        if self.response_format:
            if isinstance(self.response_format, type) and issubclass(self.response_format, BaseModel):
                raw_schema = self.response_format.model_json_schema()
                strict_schema = make_pydantic_schema_strict_json(raw_schema)
                model_params["text"] = {
                    "format": {
                        "type": "json_schema",
                        "name": self.response_format.__name__,
                        "strict": True,
                        "schema": strict_schema,
                    }
                }
            elif isinstance(self.response_format, dict):
                model_params["text"] = {"format": self.response_format}

        return {k: v for k, v in model_params.items() if v is not None}

    def _tool_to_responses(self, tool: Tool) -> Dict[str, Any]:
        """Responses-API function tool format (flat, no nested 'function' key)."""
        return {
            "type": "function",
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.parameters,
        }

    def get_client(self) -> OpenAI:
        if self.client:
            return self.client
        self.client = OpenAI(**self._get_client_params())
        return self.client

    def get_async_client(self) -> AsyncOpenAI:
        if self.async_client:
            return self.async_client
        self.async_client = AsyncOpenAI(**self._get_client_params())
        return self.async_client

    def _format_response(self, response: Optional[str]) -> Any:
        if response is None:
            return None
        json_dict = isinstance(self.response_format, dict) and self.response_format.get("type") == "json_object"
        pydantic = isinstance(self.response_format, type) and issubclass(self.response_format, BaseModel)
        if json_dict or pydantic:
            try:
                return json.loads(extract_json_str(response))
            except (json.JSONDecodeError, TypeError, ValueError):
                return response
        return response

    def _format_input(self, messages: List[Message]) -> tuple[Optional[str], List[Dict[str, Any]]]:
        """Convert gwenflow Messages into Responses-API input items.

        Returns (instructions, input_items). System messages are folded into the
        `instructions` parameter; tool calls become function_call / function_call_output items.
        """
        instructions: Optional[str] = None
        items: List[Dict[str, Any]] = []

        for message in messages:
            if message.role == "system":
                instructions = message.content if isinstance(message.content, str) else str(message.content or "")
                continue

            if message.role == "tool":
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": message.tool_call_id,
                        "output": message.content or "",
                    }
                )
                continue

            if message.role == "assistant" and message.tool_calls:
                if message.content:
                    items.append({"role": "assistant", "content": message.content})
                for tc in message.tool_calls:
                    if isinstance(tc, dict):
                        tc_id = tc["id"]
                        tc_name = tc["function"]["name"]
                        tc_args = tc["function"]["arguments"]
                    else:
                        tc_id = tc.id
                        tc_name = tc.function.name
                        tc_args = tc.function.arguments
                    args_str = tc_args if isinstance(tc_args, str) else json.dumps(tc_args)
                    items.append(
                        {
                            "type": "function_call",
                            "call_id": tc_id,
                            "name": tc_name,
                            "arguments": args_str,
                        }
                    )
                continue

            content = message.content if isinstance(message.content, str) else (message.content or "")
            items.append({"role": message.role, "content": content})

        return instructions, items

    def _build_request(self, messages: List[Message]) -> Dict[str, Any]:
        instructions, input_items = self._format_input(messages)
        payload: Dict[str, Any] = {"model": self.model, "input": input_items, **self._model_params}
        if instructions:
            payload["instructions"] = instructions
        return payload

    def _get_usage(self, usage: Any) -> Optional[RequestUsage]:
        if not usage:
            return None
        input_tokens = getattr(usage, "input_tokens", 0) or 0
        output_tokens = getattr(usage, "output_tokens", 0) or 0
        details: Dict[str, int] = {}
        cache_read_tokens = 0
        input_details = getattr(usage, "input_tokens_details", None)
        if input_details is not None:
            cache_read_tokens = getattr(input_details, "cached_tokens", 0) or 0
        output_details = getattr(usage, "output_tokens_details", None)
        if output_details is not None:
            reasoning = getattr(output_details, "reasoning_tokens", None)
            if reasoning is not None:
                details["reasoning_tokens"] = reasoning
        return RequestUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            details=details,
        )

    def _parse_response(self, response: Any) -> ModelResponse:
        parts: List[Any] = []
        finish_reason = "stop"

        for item in response.output or []:
            item_type = getattr(item, "type", None)

            if item_type == "message":
                for block in getattr(item, "content", None) or []:
                    block_type = getattr(block, "type", None)
                    if block_type in ("output_text", "text"):
                        text = getattr(block, "text", None)
                        if text:
                            parts.append(TextContent(content=text))

            elif item_type == "reasoning":
                for block in getattr(item, "summary", None) or []:
                    text = getattr(block, "text", None)
                    if text:
                        parts.append(ThinkingContent(content=text))

            elif item_type == "function_call":
                finish_reason = "tool_calls"
                parts.append(
                    ToolCall(
                        id=getattr(item, "call_id", None) or getattr(item, "id", None),
                        name=getattr(item, "name", None),
                        arguments=getattr(item, "arguments", "") or "",
                    )
                )

        model_response = ModelResponse(
            parts=parts,
            finish_reason=finish_reason,
            usage=self._get_usage(getattr(response, "usage", None)),
        )

        if self.response_format:
            model_response.parsed = self._format_response(model_response.text)

        return model_response

    @tracer.llm(name="LLM Invoke")
    def invoke(self, input: Union[str, List[Message], List[Dict[str, str]]]) -> ModelResponse:
        try:
            messages_for_model = self.input_to_message_list(input)
            response = self.get_client().responses.create(**self._build_request(messages_for_model))
        except Exception as e:
            raise RuntimeError(f"Error in calling openai API: {e}") from e
        return self._parse_response(response)

    @tracer.llm(name="LLM Async Invoke")
    async def ainvoke(self, input: Union[str, List[Message], List[Dict[str, str]]]) -> ModelResponse:
        try:
            messages_for_model = self.input_to_message_list(input)
            response = await self.get_async_client().responses.create(**self._build_request(messages_for_model))
        except Exception as e:
            raise RuntimeError(f"Error in calling openai API: {e}") from e
        return self._parse_response(response)

    @tracer.llm(name="LLM Stream")
    def stream(self, input: Union[str, List[Message], List[Dict[str, str]]]) -> Iterator[ModelResponse]:
        try:
            messages_for_model = self.input_to_message_list(input)
            request = self._build_request(messages_for_model)
            tool_calls_by_index: Dict[int, ToolCall] = {}

            with self.get_client().responses.stream(**request) as stream:
                for event in stream:
                    event_type = getattr(event, "type", None)
                    response = ModelResponse()

                    if event_type == "response.output_text.delta":
                        delta = getattr(event, "delta", "") or ""
                        if delta:
                            response.parts.append(TextContent(content=delta))
                            yield response

                    elif event_type == "response.reasoning_summary_text.delta":
                        delta = getattr(event, "delta", "") or ""
                        if delta and self.show_reasoning:
                            response.parts.append(ThinkingContent(content=delta))
                            yield response

                    elif event_type == "response.output_item.added":
                        item = getattr(event, "item", None)
                        if item is not None and getattr(item, "type", None) == "function_call":
                            tool_calls_by_index[event.output_index] = ToolCall(
                                id=getattr(item, "call_id", None) or getattr(item, "id", None),
                                name=getattr(item, "name", None),
                                arguments="",
                            )

                    elif event_type == "response.function_call_arguments.delta":
                        tc = tool_calls_by_index.get(getattr(event, "output_index", -1))
                        if tc is not None:
                            tc.arguments += getattr(event, "delta", "") or ""

                    elif event_type == "response.completed":
                        final = getattr(event, "response", None)
                        if tool_calls_by_index:
                            response.parts.extend(tool_calls_by_index.values())
                            response.finish_reason = "tool_calls"
                        if final is not None:
                            response.usage = self._get_usage(getattr(final, "usage", None))
                        if response.parts or response.usage:
                            yield response

        except Exception as e:
            raise RuntimeError(f"Error in calling openai API: {e}") from e

    @tracer.llm(name="LLM Async Astream")
    async def astream(self, input: Union[str, List[Message], List[Dict[str, str]]]) -> AsyncIterator[ModelResponse]:
        try:
            messages_for_model = self.input_to_message_list(input)
            request = self._build_request(messages_for_model)
            tool_calls_by_index: Dict[int, ToolCall] = {}

            async with self.get_async_client().responses.stream(**request) as stream:
                async for event in stream:
                    event_type = getattr(event, "type", None)
                    response = ModelResponse()

                    if event_type == "response.output_text.delta":
                        delta = getattr(event, "delta", "") or ""
                        if delta:
                            response.parts.append(TextContent(content=delta))
                            yield response

                    elif event_type == "response.reasoning_summary_text.delta":
                        delta = getattr(event, "delta", "") or ""
                        if delta and self.show_reasoning:
                            response.parts.append(ThinkingContent(content=delta))
                            yield response

                    elif event_type == "response.output_item.added":
                        item = getattr(event, "item", None)
                        if item is not None and getattr(item, "type", None) == "function_call":
                            tool_calls_by_index[event.output_index] = ToolCall(
                                id=getattr(item, "call_id", None) or getattr(item, "id", None),
                                name=getattr(item, "name", None),
                                arguments="",
                            )

                    elif event_type == "response.function_call_arguments.delta":
                        tc = tool_calls_by_index.get(getattr(event, "output_index", -1))
                        if tc is not None:
                            tc.arguments += getattr(event, "delta", "") or ""

                    elif event_type == "response.completed":
                        final = getattr(event, "response", None)
                        if tool_calls_by_index:
                            response.parts.extend(tool_calls_by_index.values())
                            response.finish_reason = "tool_calls"
                        if final is not None:
                            response.usage = self._get_usage(getattr(final, "usage", None))
                        if response.parts or response.usage:
                            yield response

        except Exception as e:
            raise RuntimeError(f"Error in calling openai API: {e}") from e
