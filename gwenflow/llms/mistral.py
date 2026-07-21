"""Mistral AI provider using the native `mistralai` SDK.

Supports the full ChatBase contract: invoke / ainvoke / stream / astream, tool
calls, structured output, multi-modal input (text / image / PDF), and thinking
output from Magistral reasoning models (`ThinkChunk`).
"""

import json
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Type, Union

from mistralai.client import Mistral as MistralClient
from mistralai.client.models import (
    AssistantMessage,
    DocumentURLChunk,
    ImageURLChunk,
    SystemMessage,
    TextChunk,
    ThinkChunk,
    ToolMessage,
    UserMessage,
)
from pydantic import BaseModel

from gwenflow.llms.base import ChatBase
from gwenflow.telemetry import tracer
from gwenflow.tools import Tool
from gwenflow.types import (
    FileContent,
    ImageContent,
    Message,
    ModelResponse,
    RequestUsage,
    TextContent,
    ThinkingContent,
    ToolCall,
)
from gwenflow.utils import extract_json_str, make_pydantic_schema_strict_json


@dataclass(kw_only=True)
class ChatMistral(ChatBase):
    model: str = "mistral-small-2603"

    # model parameters
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    random_seed: Optional[int] = None
    response_format: Optional[Union[Dict[str, Any], Type[BaseModel]]] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None

    # client (mistralai uses one class for sync + async)
    client: Optional[MistralClient] = None

    # client parameters
    api_key: Optional[str] = None
    server_url: Optional[str] = None
    timeout_ms: Optional[int] = None

    # internal
    system_prompt: Optional[str] = None

    def _get_client_params(self) -> Dict[str, Any]:
        api_key = self.api_key
        if api_key is None:
            api_key = os.environ.get("MISTRAL_API_KEY")
        if api_key is None:
            raise ValueError(
                "The api_key client option must be set either by passing api_key to the client or by setting the MISTRAL_API_KEY environment variable"
            )

        client_params: Dict[str, Any] = {"api_key": api_key}
        if self.server_url is not None:
            client_params["server_url"] = self.server_url
        if self.timeout_ms is not None:
            client_params["timeout_ms"] = self.timeout_ms
        return client_params

    def get_client(self) -> MistralClient:
        if self.client is None:
            self.client = MistralClient(**self._get_client_params())
        return self.client

    # Mistral SDK exposes async methods on the same client, so there's no
    # separate async_client to manage. Match ChatBase by aliasing.
    get_async_client = get_client

    async def aclose(self) -> None:
        """Mistral SDK doesn't expose an async-pool close hook today; no-op."""
        return None

    @property
    def _model_params(self) -> Dict[str, Any]:
        model_params: Dict[str, Any] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "stop": self.stop,
            "random_seed": self.random_seed,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
        }

        if self.tools and self.tool_type == "fncall":
            model_params["tools"] = [self._tool_to_mistral(tool) for tool in self.tools]
            model_params["tool_choice"] = self.tool_choice or "auto"

        if self.response_format:
            if isinstance(self.response_format, type) and issubclass(self.response_format, BaseModel):
                raw_schema = self.response_format.model_json_schema()
                strict_schema = make_pydantic_schema_strict_json(raw_schema)
                model_params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": self.response_format.__name__,
                        "strict": True,
                        "schema": strict_schema,
                    },
                }
            else:
                model_params["response_format"] = self.response_format

        return {k: v for k, v in model_params.items() if v is not None}

    def _tool_to_mistral(self, tool: Tool) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.parameters,
            },
        }

    @staticmethod
    def _part_to_mistral(part: Any) -> Any:
        """Translate a MessageContent part to a Mistral content chunk."""
        if isinstance(part, dict):
            return part  # raw provider dict
        if isinstance(part, TextContent):
            return TextChunk(text=part.content)
        if isinstance(part, ImageContent):
            if part.url:
                return ImageURLChunk(image_url=part.url)
            if part.data:
                return ImageURLChunk(image_url=f"data:{part.media_type};base64,{part.data}")
            raise ValueError("ImageContent requires either `url` or `data`.")
        if isinstance(part, FileContent):
            if part.url:
                return DocumentURLChunk(document_url=part.url)
            raise ValueError(
                "Mistral document chunks require a hosted URL; upload first or pass `url=` "
                "(inline base64 PDFs are not supported by the Mistral SDK)."
            )
        # Fallback to text representation for anything else
        return TextChunk(text=str(part))

    def _format_messages(self, messages: List[Message]) -> List[Any]:
        """Convert gwenflow Messages to Mistral SDK message objects."""
        formatted: List[Any] = []
        for message in messages:
            if message.role == "system":
                formatted.append(SystemMessage(content=message.content or ""))
                continue

            if message.role == "tool":
                formatted.append(
                    ToolMessage(
                        content=message.content or "",
                        tool_call_id=message.tool_call_id,
                        name=message.name,
                    )
                )
                continue

            content: Any
            if isinstance(message.content, list):
                content = [self._part_to_mistral(p) for p in message.content]
            else:
                content = message.content or ""

            if message.role == "assistant":
                # Replay thinking parts as ThinkChunks (helps the model maintain
                # state in multi-turn Magistral conversations).
                if message.thinking_parts:
                    chunks: List[Any] = []
                    for tp in message.thinking_parts:
                        chunks.append(ThinkChunk(thinking=[{"type": "text", "text": tp.content}], closed=True))
                    if isinstance(content, list):
                        chunks.extend(content)
                    elif content:
                        chunks.append(TextChunk(text=content))
                    content = chunks

                tool_calls = None
                if message.tool_calls:
                    tool_calls = []
                    for tc in message.tool_calls:
                        tc_id = tc["id"] if isinstance(tc, dict) else tc.id
                        tc_name = tc["function"]["name"] if isinstance(tc, dict) else tc.function.name
                        tc_args = tc["function"]["arguments"] if isinstance(tc, dict) else tc.function.arguments
                        tool_calls.append(
                            {
                                "id": tc_id,
                                "type": "function",
                                "function": {"name": tc_name, "arguments": tc_args},
                            }
                        )
                formatted.append(AssistantMessage(content=content, tool_calls=tool_calls))
            else:  # user
                formatted.append(UserMessage(content=content))

        return formatted

    def _get_usage(self, usage) -> Optional[RequestUsage]:
        if usage is None:
            return None
        return RequestUsage(
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )

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

    @staticmethod
    def _think_chunk_text(chunk: ThinkChunk) -> str:
        """ThinkChunk.thinking is a list of items each with a `text` field."""
        parts: List[str] = []
        for item in chunk.thinking or []:
            text = item.get("text") if isinstance(item, dict) else getattr(item, "text", None)
            if text:
                parts.append(text)
        return "".join(parts)

    def _parse_response(self, response) -> ModelResponse:
        parts: List[Any] = []
        finish_reason = "stop"

        for choice in response.choices:
            msg = choice.message
            content = msg.content

            if isinstance(content, list):
                for chunk in content:
                    if isinstance(chunk, ThinkChunk):
                        text = self._think_chunk_text(chunk)
                        if text:
                            parts.append(ThinkingContent(content=text))
                    elif isinstance(chunk, TextChunk):
                        parts.append(TextContent(content=chunk.text))
                    else:
                        # Unknown chunk type — capture as text representation
                        parts.append(TextContent(content=str(chunk)))
            elif isinstance(content, str) and content:
                parts.append(TextContent(content=content))

            if msg.tool_calls:
                for tc in msg.tool_calls:
                    arguments = tc.function.arguments
                    if not isinstance(arguments, str):
                        arguments = json.dumps(arguments)
                    parts.append(ToolCall(id=tc.id, name=tc.function.name, arguments=arguments))

            if choice.finish_reason == "tool_calls":
                finish_reason = "tool_calls"

        model_response = ModelResponse(
            parts=parts,
            finish_reason=finish_reason,
            usage=self._get_usage(getattr(response, "usage", None)),
        )

        if self.response_format:
            model_response.parsed = self._format_response(model_response.text, response_format=self.response_format)

        return model_response

    def _build_request(self, messages: List[Message]) -> Dict[str, Any]:
        return {
            "model": self.model,
            "messages": self._format_messages(messages),
            **self._model_params,
        }

    @tracer.llm(name="LLM Invoke")
    def invoke(self, input: Union[str, List[Message], List[Dict[str, str]]]) -> ModelResponse:
        try:
            messages_for_model = self.input_to_message_list(input)
            response = self.get_client().chat.complete(**self._build_request(messages_for_model))
        except Exception as e:
            raise RuntimeError(f"Error in calling mistral API: {e}") from e
        return self._parse_response(response)

    @tracer.llm(name="LLM Async Invoke")
    async def ainvoke(self, input: Union[str, List[Message], List[Dict[str, str]]]) -> ModelResponse:
        try:
            messages_for_model = self.input_to_message_list(input)
            response = await self.get_client().chat.complete_async(**self._build_request(messages_for_model))
        except Exception as e:
            raise RuntimeError(f"Error in calling mistral API: {e}") from e
        return self._parse_response(response)

    def _process_stream_chunk(
        self,
        event,
        full_tool_calls: List[ToolCall],
    ) -> ModelResponse:
        response = ModelResponse()
        chunk = event.data
        if not chunk.choices:
            response.usage = self._get_usage(getattr(chunk, "usage", None))
            return response

        delta = chunk.choices[0].delta
        finish_reason = chunk.choices[0].finish_reason

        content = delta.content
        if isinstance(content, list):
            for c in content:
                if isinstance(c, ThinkChunk):
                    text = self._think_chunk_text(c)
                    if text:
                        response.parts.append(ThinkingContent(content=text))
                elif isinstance(c, TextChunk):
                    response.parts.append(TextContent(content=c.text))
        elif isinstance(content, str) and content:
            response.parts.append(TextContent(content=content))

        if delta.tool_calls:
            for tc in delta.tool_calls:
                tc_id = getattr(tc, "id", None)
                if full_tool_calls and (not tc_id or tc_id == full_tool_calls[-1].id):
                    if tc.function.name:
                        full_tool_calls[-1].name += tc.function.name
                    if tc.function.arguments:
                        full_tool_calls[-1].arguments = (full_tool_calls[-1].arguments or "") + tc.function.arguments
                else:
                    full_tool_calls.append(
                        ToolCall(id=tc_id, name=tc.function.name, arguments=tc.function.arguments or "")
                    )

        if finish_reason == "tool_calls" and full_tool_calls:
            response.parts.extend(full_tool_calls)
            response.finish_reason = "tool_calls"

        if chunk.usage is not None:
            response.usage = self._get_usage(chunk.usage)

        return response

    @tracer.llm(name="LLM Stream")
    def stream(self, input: Union[str, List[Message], List[Dict[str, str]]]) -> Iterator[ModelResponse]:
        try:
            messages_for_model = self.input_to_message_list(input)
            events = self.get_client().chat.stream(**self._build_request(messages_for_model))
            full_tool_calls: List[ToolCall] = []
            for event in events:
                response = self._process_stream_chunk(event, full_tool_calls)
                if response.parts or response.usage:
                    yield response
        except Exception as e:
            raise RuntimeError(f"Error in calling mistral API: {e}") from e

    @tracer.llm(name="LLM Async Astream")
    async def astream(self, input: Union[str, List[Message], List[Dict[str, str]]]) -> AsyncIterator[ModelResponse]:
        try:
            messages_for_model = self.input_to_message_list(input)
            events = await self.get_client().chat.stream_async(**self._build_request(messages_for_model))
            full_tool_calls: List[ToolCall] = []
            async for event in events:
                response = self._process_stream_chunk(event, full_tool_calls)
                if response.parts or response.usage:
                    yield response
        except Exception as e:
            raise RuntimeError(f"Error in calling mistral API: {e}") from e

    def get_thinking_parts(self, response: ModelResponse) -> Optional[List[ThinkingContent]]:
        """Coalesce streamed ThinkChunk deltas into one ThinkingContent per block.

        Magistral models emit thinking as a separate stream of ThinkChunks.
        Including them on echo-back helps the model continue its reasoning in
        multi-turn conversations.
        """
        thinking = [p for p in response.parts if isinstance(p, ThinkingContent)]
        if not thinking:
            return None
        merged = "".join(p.content for p in thinking)
        return [ThinkingContent(content=merged)]
