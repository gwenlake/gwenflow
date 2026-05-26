"""Google Gemini provider using the native `google-genai` SDK.

Supports the full ChatBase contract: invoke / ainvoke / stream / astream, tool
calls, structured output, multi-modal input (image / audio / PDF), and Gemini
2.5's thinking output (`Part.thought=True` with `thought_signature`).

The Google API has a different shape from OpenAI-compat: roles are
`user`/`model`, system prompts are config (not a content), and parts are typed
records rather than wire-shaped dicts.
"""

import base64
import json
import logging
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Type, Union

from google import genai
from google.genai import types as gt
from pydantic import BaseModel

from gwenflow.llms.base import ChatBase
from gwenflow.telemetry import tracer
from gwenflow.tools import Tool
from gwenflow.types import (
    AudioContent,
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

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class ChatGoogle(ChatBase):
    model: str = "gemini-2.5-flash"

    # model parameters
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_output_tokens: Optional[int] = None
    stop_sequences: Optional[List[str]] = None
    seed: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    response_format: Optional[Union[Dict[str, Any], Type[BaseModel]]] = None

    # thinking config — pass e.g. {"include_thoughts": True, "thinking_budget": 1024}
    thinking: Optional[Dict[str, Any]] = None

    # client (genai.Client wraps both sync and async via .aio)
    client: Optional[genai.Client] = None

    # client parameters
    api_key: Optional[str] = None
    vertexai: Optional[bool] = None
    project: Optional[str] = None
    location: Optional[str] = None
    timeout_ms: Optional[int] = None

    # internal
    system_prompt: Optional[str] = None

    def _get_client_params(self) -> Dict[str, Any]:
        api_key = self.api_key
        if api_key is None:
            api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if api_key is None and not self.vertexai:
            raise ValueError("Set GOOGLE_API_KEY (or GEMINI_API_KEY) in env, or pass api_key=, or use vertexai=True.")

        client_params: Dict[str, Any] = {}
        if api_key is not None:
            client_params["api_key"] = api_key
        if self.vertexai is not None:
            client_params["vertexai"] = self.vertexai
        if self.project is not None:
            client_params["project"] = self.project
        if self.location is not None:
            client_params["location"] = self.location
        if self.timeout_ms is not None:
            client_params["http_options"] = gt.HttpOptions(timeout=self.timeout_ms)
        return client_params

    def get_client(self) -> genai.Client:
        if self.client is None:
            self.client = genai.Client(**self._get_client_params())
        return self.client

    # genai.Client exposes both sync and async via .aio — same client serves both
    get_async_client = get_client

    async def aclose(self) -> None:
        return None

    def _build_config(self) -> gt.GenerateContentConfig:
        config_kwargs: Dict[str, Any] = {}
        if self.system_prompt:
            config_kwargs["system_instruction"] = self.system_prompt
        if self.temperature is not None:
            config_kwargs["temperature"] = self.temperature
        if self.top_p is not None:
            config_kwargs["top_p"] = self.top_p
        if self.top_k is not None:
            config_kwargs["top_k"] = self.top_k
        if self.max_output_tokens is not None:
            config_kwargs["max_output_tokens"] = self.max_output_tokens
        if self.stop_sequences is not None:
            config_kwargs["stop_sequences"] = self.stop_sequences
        if self.seed is not None:
            config_kwargs["seed"] = self.seed
        if self.presence_penalty is not None:
            config_kwargs["presence_penalty"] = self.presence_penalty
        if self.frequency_penalty is not None:
            config_kwargs["frequency_penalty"] = self.frequency_penalty

        if self.tools and self.tool_type == "fncall":
            declarations = [self._tool_to_google(t) for t in self.tools]
            config_kwargs["tools"] = [gt.Tool(function_declarations=declarations)]
            if self.tool_choice:
                mode = {"required": "ANY", "auto": "AUTO", "none": "NONE"}.get(self.tool_choice, "AUTO")
                config_kwargs["tool_config"] = gt.ToolConfig(
                    function_calling_config=gt.FunctionCallingConfig(mode=mode)
                )

        if self.response_format:
            if isinstance(self.response_format, type) and issubclass(self.response_format, BaseModel):
                raw_schema = self.response_format.model_json_schema()
                strict_schema = make_pydantic_schema_strict_json(raw_schema)
                config_kwargs["response_mime_type"] = "application/json"
                config_kwargs["response_json_schema"] = strict_schema
            elif isinstance(self.response_format, dict):
                if self.response_format.get("type") == "json_object":
                    config_kwargs["response_mime_type"] = "application/json"

        if self.thinking:
            config_kwargs["thinking_config"] = gt.ThinkingConfig(**self.thinking)

        return gt.GenerateContentConfig(**config_kwargs)

    def _tool_to_google(self, tool: Tool) -> gt.FunctionDeclaration:
        return gt.FunctionDeclaration(
            name=tool.name,
            description=tool.description or "",
            parameters_json_schema=tool.parameters,
        )

    @staticmethod
    def _part_to_google(part: Any) -> gt.Part:
        """Translate a MessageContent part to a Gemini Part."""
        if isinstance(part, gt.Part):
            return part
        if isinstance(part, dict):
            # Allow raw provider-shaped dicts to flow through.
            return gt.Part(**part)
        if isinstance(part, TextContent):
            return gt.Part(text=part.content)
        if isinstance(part, ImageContent):
            if part.url:
                return gt.Part(file_data=gt.FileData(file_uri=part.url, mime_type=part.media_type))
            if part.data:
                return gt.Part(inline_data=gt.Blob(mime_type=part.media_type, data=base64.b64decode(part.data)))
            raise ValueError("ImageContent requires either `url` or `data`.")
        if isinstance(part, AudioContent):
            mime = f"audio/{part.format}" if part.format != "pcm16" else "audio/pcm"
            return gt.Part(inline_data=gt.Blob(mime_type=mime, data=base64.b64decode(part.data)))
        if isinstance(part, FileContent):
            if part.url:
                return gt.Part(file_data=gt.FileData(file_uri=part.url, mime_type=part.media_type))
            if part.data:
                return gt.Part(inline_data=gt.Blob(mime_type=part.media_type, data=base64.b64decode(part.data)))
            raise ValueError("Gemini requires either an uploaded file URI (via Files API) or inline base64 data.")
        return gt.Part(text=str(part))

    def _format_messages(self, messages: List[Message]) -> List[gt.Content]:
        """Convert gwenflow Messages to Gemini Content objects.

        Sets `self.system_prompt` as a side effect — Gemini wants the system
        instruction on `GenerateContentConfig`, not in the contents list.
        """
        self.system_prompt = None
        formatted: List[gt.Content] = []

        for message in messages:
            if message.role == "system":
                self.system_prompt = message.content if isinstance(message.content, str) else ""
                continue

            if message.role == "tool":
                # tool result → user role, function_response part
                try:
                    response_data = (
                        json.loads(message.content) if isinstance(message.content, str) else (message.content or {})
                    )
                except (json.JSONDecodeError, TypeError):
                    response_data = {"result": message.content}
                formatted.append(
                    gt.Content(
                        role="user",
                        parts=[
                            gt.Part(
                                function_response=gt.FunctionResponse(
                                    name=message.name or "",
                                    response=response_data
                                    if isinstance(response_data, dict)
                                    else {"result": response_data},
                                )
                            )
                        ],
                    )
                )
                continue

            role = "model" if message.role == "assistant" else "user"
            parts: List[gt.Part] = []

            # Replay thinking parts (with signatures) for tool-use continuity
            if message.role == "assistant" and message.thinking_parts:
                for tp in message.thinking_parts:
                    extra: Dict[str, Any] = {"text": tp.content, "thought": True}
                    if tp.extra and tp.extra.get("thought_signature"):
                        sig = tp.extra["thought_signature"]
                        if isinstance(sig, str):
                            try:
                                sig = base64.b64decode(sig)
                            except Exception:
                                pass
                        extra["thought_signature"] = sig
                    parts.append(gt.Part(**extra))

            if isinstance(message.content, list):
                parts.extend(self._part_to_google(p) for p in message.content)
            elif isinstance(message.content, str) and message.content:
                parts.append(gt.Part(text=message.content))

            if message.role == "assistant" and message.tool_calls:
                for tc in message.tool_calls:
                    tc_id = tc["id"] if isinstance(tc, dict) else tc.id
                    tc_name = tc["function"]["name"] if isinstance(tc, dict) else tc.function.name
                    tc_args = tc["function"]["arguments"] if isinstance(tc, dict) else tc.function.arguments
                    try:
                        args_dict = json.loads(tc_args) if isinstance(tc_args, str) else (tc_args or {})
                    except (json.JSONDecodeError, TypeError):
                        args_dict = {}
                    parts.append(gt.Part(function_call=gt.FunctionCall(name=tc_name, args=args_dict, id=tc_id)))

            if parts:
                formatted.append(gt.Content(role=role, parts=parts))

        return formatted

    def _get_usage(self, usage_metadata) -> Optional[RequestUsage]:
        if usage_metadata is None:
            return None
        details: Dict[str, int] = {}
        thoughts = getattr(usage_metadata, "thoughts_token_count", None)
        if thoughts:
            details["reasoning_tokens"] = thoughts
        return RequestUsage(
            input_tokens=getattr(usage_metadata, "prompt_token_count", 0) or 0,
            output_tokens=getattr(usage_metadata, "candidates_token_count", 0) or 0,
            cache_read_tokens=getattr(usage_metadata, "cached_content_token_count", 0) or 0,
            details=details,
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
    def _parts_from_candidate(candidate) -> List[Any]:
        parts: List[Any] = []
        if not candidate.content or not candidate.content.parts:
            return parts
        for part in candidate.content.parts:
            if getattr(part, "thought", False) and part.text:
                extra: Dict[str, Any] = {}
                sig = getattr(part, "thought_signature", None)
                if sig:
                    extra["thought_signature"] = (
                        base64.b64encode(sig).decode("ascii") if isinstance(sig, bytes) else sig
                    )
                parts.append(ThinkingContent(content=part.text, extra=extra))
            elif part.text:
                parts.append(TextContent(content=part.text))
            elif part.function_call:
                fc = part.function_call
                args = fc.args if isinstance(fc.args, str) else json.dumps(fc.args or {})
                parts.append(ToolCall(id=getattr(fc, "id", None) or fc.name, name=fc.name, arguments=args))
            elif part.inline_data and part.inline_data.data:
                mime = part.inline_data.mime_type or "application/octet-stream"
                data_b64 = base64.b64encode(part.inline_data.data).decode("ascii")
                if mime.startswith("image/"):
                    parts.append(ImageContent(data=data_b64, media_type=mime))
                elif mime.startswith("audio/"):
                    fmt = mime.split("/", 1)[1]
                    parts.append(AudioContent(data=data_b64, format=fmt))
                else:
                    parts.append(TextContent(content=f"[inline {mime}, {len(part.inline_data.data)} bytes]"))
        return parts

    def _parse_response(self, response) -> ModelResponse:
        parts: List[Any] = []
        finish_reason = "stop"

        for candidate in response.candidates or []:
            parts.extend(self._parts_from_candidate(candidate))
            fr = getattr(candidate, "finish_reason", None)
            if fr and str(fr).lower() == "tool_use":
                finish_reason = "tool_calls"

        # If we have tool calls, prefer that finish reason
        if any(isinstance(p, ToolCall) for p in parts) and finish_reason == "stop":
            finish_reason = "tool_calls"

        model_response = ModelResponse(
            parts=parts,
            finish_reason=finish_reason,
            usage=self._get_usage(getattr(response, "usage_metadata", None)),
        )

        if self.response_format:
            model_response.parsed = self._format_response(model_response.text, response_format=self.response_format)

        return model_response

    def _build_request(self, messages: List[Message]) -> Dict[str, Any]:
        contents = self._format_messages(messages)
        return {
            "model": self.model,
            "contents": contents,
            "config": self._build_config(),
        }

    @tracer.llm(name="LLM Invoke")
    def invoke(self, input: Union[str, List[Message], List[Dict[str, str]]]) -> ModelResponse:
        try:
            messages_for_model = self.input_to_message_list(input)
            response = self.get_client().models.generate_content(**self._build_request(messages_for_model))
        except Exception as e:
            raise RuntimeError(f"Error in calling google API: {e}") from e
        return self._parse_response(response)

    @tracer.llm(name="LLM Async Invoke")
    async def ainvoke(self, input: Union[str, List[Message], List[Dict[str, str]]]) -> ModelResponse:
        try:
            messages_for_model = self.input_to_message_list(input)
            response = await self.get_client().aio.models.generate_content(**self._build_request(messages_for_model))
        except Exception as e:
            raise RuntimeError(f"Error in calling google API: {e}") from e
        return self._parse_response(response)

    def _stream_chunk_to_response(self, chunk) -> ModelResponse:
        response = ModelResponse()
        for candidate in chunk.candidates or []:
            response.parts.extend(self._parts_from_candidate(candidate))
        usage = self._get_usage(getattr(chunk, "usage_metadata", None))
        if usage is not None:
            response.usage = usage
        return response

    @tracer.llm(name="LLM Stream")
    def stream(self, input: Union[str, List[Message], List[Dict[str, str]]]) -> Iterator[ModelResponse]:
        try:
            messages_for_model = self.input_to_message_list(input)
            stream = self.get_client().models.generate_content_stream(**self._build_request(messages_for_model))
            for chunk in stream:
                response = self._stream_chunk_to_response(chunk)
                if response.parts or response.usage:
                    yield response
        except Exception as e:
            raise RuntimeError(f"Error in calling google API: {e}") from e

    @tracer.llm(name="LLM Async Astream")
    async def astream(self, input: Union[str, List[Message], List[Dict[str, str]]]) -> AsyncIterator[ModelResponse]:
        try:
            messages_for_model = self.input_to_message_list(input)
            stream = await self.get_client().aio.models.generate_content_stream(
                **self._build_request(messages_for_model)
            )
            async for chunk in stream:
                response = self._stream_chunk_to_response(chunk)
                if response.parts or response.usage:
                    yield response
        except Exception as e:
            raise RuntimeError(f"Error in calling google API: {e}") from e

    def get_thinking_parts(self, response: ModelResponse) -> Optional[List[ThinkingContent]]:
        """Collect ThinkingContent parts (with `thought_signature` for echo-back).

        Gemini 2.5 emits thinking as `Part(thought=True)` parts. For tool-use
        continuity the `thought_signature` must be replayed verbatim; we store
        it (base64) in `ThinkingContent.extra` and `_format_messages` replays it.
        """
        thinking = [p for p in response.parts if isinstance(p, ThinkingContent)]
        return thinking or None
