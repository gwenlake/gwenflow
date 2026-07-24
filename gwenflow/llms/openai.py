import asyncio
import json
import os
import re
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple, Type, Union

from openai import AsyncOpenAI, OpenAI
from openai.types import Batch
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import ChoiceDelta
from pydantic import BaseModel

from gwenflow.llms.base import ChatBase
from gwenflow.logger import logger
from gwenflow.telemetry import tracer
from gwenflow.tools import Tool
from gwenflow.types import (
    AudioContent,
    Message,
    ModelResponse,
    RequestUsage,
    TextContent,
    ThinkingContent,
    ToolCall,
)
from gwenflow.utils import extract_json_str, make_pydantic_schema_strict_json

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_THINK_BLOCK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def _split_think_tags(content: str) -> Tuple[Optional[str], str]:
    """Pull <think>...</think> blocks out of a full string.

    Returns (thinking_or_None, remaining_text). Used as a fallback for local
    models (Gemma 3, Qwen3, DeepSeek-R1 distills) whose servers emit reasoning
    inline rather than in a dedicated `reasoning_content` field.
    """
    blocks = _THINK_BLOCK_RE.findall(content)
    if not blocks:
        return None, content
    remaining = _THINK_BLOCK_RE.sub("", content).strip()
    thinking = "\n\n".join(b.strip() for b in blocks)
    return thinking, remaining


class _ThinkTagStreamExtractor:
    """Streaming state machine that splits <think>...</think> from text chunks.

    Tags can straddle chunk boundaries, so we keep a small tail buffer
    (len(tag) - 1 chars) before emitting, to avoid mis-classifying a partial tag.
    """

    def __init__(self) -> None:
        self._buf = ""
        self._in_think = False

    def feed(self, text: str) -> List[Tuple[str, str]]:
        self._buf += text
        out: List[Tuple[str, str]] = []
        while self._buf:
            if self._in_think:
                idx = self._buf.find(_THINK_CLOSE)
                if idx == -1:
                    safe = len(self._buf) - (len(_THINK_CLOSE) - 1)
                    if safe > 0:
                        out.append(("thinking", self._buf[:safe]))
                        self._buf = self._buf[safe:]
                    break
                if idx > 0:
                    out.append(("thinking", self._buf[:idx]))
                self._buf = self._buf[idx + len(_THINK_CLOSE) :]
                self._in_think = False
            else:
                idx = self._buf.find(_THINK_OPEN)
                if idx == -1:
                    safe = len(self._buf) - (len(_THINK_OPEN) - 1)
                    if safe > 0:
                        out.append(("text", self._buf[:safe]))
                        self._buf = self._buf[safe:]
                    break
                if idx > 0:
                    out.append(("text", self._buf[:idx]))
                self._buf = self._buf[idx + len(_THINK_OPEN) :]
                self._in_think = True
        return out

    def flush(self) -> List[Tuple[str, str]]:
        if not self._buf:
            return []
        kind = "thinking" if self._in_think else "text"
        out = [(kind, self._buf)]
        self._buf = ""
        return out


def _audio_delta_to_part(audio: Any) -> Optional[AudioContent]:
    """Build an AudioContent from a streaming `delta.audio` chunk (gpt-4o-audio).

    Returns None when the delta carries no payload. The chunk shape is roughly
    {id?, data?, transcript?, expires_at?}; we surface the base64 data and/or
    transcript fragment and stash id/expires_at in `extra` so downstream
    consumers can correlate chunks belonging to the same audio response.
    """
    if audio is None:
        return None
    data = getattr(audio, "data", None) or ""
    transcript = getattr(audio, "transcript", None)
    if not data and not transcript:
        return None
    extra: Dict[str, Any] = {}
    for attr in ("id", "expires_at"):
        val = getattr(audio, attr, None)
        if val is not None:
            extra[attr] = val
    return AudioContent(data=data, transcript=transcript, extra=extra)


def response_to_openai_dict(response: ModelResponse) -> Dict[str, Any]:
    """Serialize a ModelResponse to an OpenAI ChatCompletion-shaped dict.

    Thinking goes into `message.reasoning` (the field used by gpt-5 / o-series
    via Chat Completions). For the DeepSeek `reasoning_content` variant, use
    `response_to_deepseek_dict` from `gwenflow.llms.deepseek`.
    """
    message: Dict[str, Any] = {"role": "assistant", "content": response.text}

    if response.thinking is not None:
        message["reasoning"] = response.thinking

    if response.tool_calls:
        message["tool_calls"] = [tc.to_message_dict() for tc in response.tool_calls]

    completion: Dict[str, Any] = {
        "id": response.id,
        "object": "chat.completion",
        "created": int(response.created_at.timestamp()),
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": response.finish_reason,
            }
        ],
    }

    if response.usage is not None:
        usage_dict: Dict[str, Any] = {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.total_tokens,
        }
        if response.usage.details:
            usage_dict["completion_tokens_details"] = response.usage.details
        completion["usage"] = usage_dict

    return completion


_BATCH_TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}


@dataclass
class BatchResultItem:
    custom_id: str
    response: Optional[ModelResponse] = None
    error: Optional[Dict[str, Any]] = None


@dataclass(kw_only=True)
class ChatOpenAI(ChatBase):
    model: str = "gpt-5-mini"

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
    response_format: Optional[Union[Dict[str, Any], Type[BaseModel]]] = None
    seed: Optional[int] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    prompt_cache_key: Optional[str] = None

    # reasoning / thinking
    reasoning_effort: Optional[str] = None
    """For gpt-5 / o-series reasoning models: 'minimal' | 'low' | 'medium' | 'high'."""

    extract_think_tags: bool = False
    """If True, extract inline <think>...</think> blocks from content into thinking parts.
    Use for local models served via OpenAI-compatible APIs (Gemma 3, Qwen3,
    DeepSeek-R1 distills) when the server doesn't expose a `reasoning_content` field."""

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
            organization = os.environ.get("OPENAI_ORG_ID")

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
            "seed": self.seed,
            "logprobs": self.logprobs,
            "top_logprobs": self.top_logprobs,
            "reasoning_effort": self.reasoning_effort,
            "prompt_cache_key": self.prompt_cache_key,
        }

        if self.tools and self.tool_type == "fncall":
            model_params["tools"] = [self._tool_to_openai(tool) for tool in self.tools]
            model_params["tool_choice"] = self.tool_choice or "auto"

        if self.response_format:
            if isinstance(self.response_format, type) and issubclass(self.response_format, BaseModel):
                raw_schema = self.response_format.model_json_schema()
                strict_schema = make_pydantic_schema_strict_json(raw_schema)

                model_params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": self.response_format.__name__, "strict": True, "schema": strict_schema},
                }
            else:
                model_params["response_format"] = self.response_format

        model_params = {k: v for k, v in model_params.items() if v is not None}

        return model_params

    def _tool_to_openai(self, tool: Tool) -> dict:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.parameters,
            },
        }

    def get_client(self) -> OpenAI:
        if self.client:
            return self.client
        client_params = self._get_client_params()
        self.client = OpenAI(**client_params)
        return self.client

    def get_async_client(self) -> AsyncOpenAI:
        if self.async_client:
            return self.async_client
        client_params = self._get_client_params()
        self.async_client = AsyncOpenAI(**client_params)
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

    def _format_message(self, message: Message) -> Dict[str, Any]:
        """Format a message into the format expected by OpenAI."""
        return message.to_openai()

    def _get_usage(self, completion: Union[ChatCompletion, ChatCompletionChunk]) -> RequestUsage:
        if not hasattr(completion, "usage"):
            return None
        if not completion.usage:
            return None
        details = {}
        if hasattr(completion.usage, "completion_tokens_details"):
            if completion.usage.completion_tokens_details:
                details = completion.usage.completion_tokens_details.model_dump()
        cache_read_tokens = 0
        prompt_details = getattr(completion.usage, "prompt_tokens_details", None)
        if prompt_details is not None:
            cache_read_tokens = getattr(prompt_details, "cached_tokens", 0) or 0
        return RequestUsage(
            input_tokens=completion.usage.prompt_tokens,
            output_tokens=completion.usage.completion_tokens,
            cache_read_tokens=cache_read_tokens,
            details=details,
        )

    def _parse_response(self, completion: ChatCompletion) -> ModelResponse:
        parts = []
        for block in completion.choices:
            msg = block.message

            reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)

            content = msg.content

            if self.extract_think_tags and content:
                tag_thinking, content = _split_think_tags(content)
                if tag_thinking and not reasoning:
                    reasoning = tag_thinking

            if reasoning:
                parts.append(ThinkingContent(content=reasoning))
            if content:
                parts.append(TextContent(content=content))

            audio = getattr(msg, "audio", None)
            if audio is not None and getattr(audio, "data", None):
                extra: Dict[str, Any] = {}
                for attr in ("id", "expires_at"):
                    val = getattr(audio, attr, None)
                    if val is not None:
                        extra[attr] = val
                parts.append(
                    AudioContent(
                        data=audio.data,
                        format=getattr(audio, "format", "wav") or "wav",
                        transcript=getattr(audio, "transcript", None),
                        extra=extra,
                    )
                )

            tool_calls = msg.tool_calls
            if tool_calls is not None and len(tool_calls) > 0:
                try:
                    tool_calls = [
                        ToolCall(id=t.id, name=t.function.name, arguments=t.function.arguments) for t in tool_calls
                    ]
                    parts.extend(tool_calls)
                except Exception as e:
                    logger.warning(f"Error processing tool calls: {e}")

        model_response = ModelResponse(
            parts=parts,
            finish_reason="stop",
            usage=self._get_usage(completion),
        )

        if self.response_format:
            model_response.parsed = self._format_response(model_response.text, response_format=self.response_format)

        return model_response

    @tracer.llm(name="LLM Invoke")
    def invoke(self, input: Union[str, List[Message], List[Dict[str, str]]]) -> ModelResponse:
        try:
            messages_for_model = self.input_to_message_list(input)
            response: ChatCompletion = self.get_client().chat.completions.create(
                model=self.model,
                messages=[self._format_message(m) for m in messages_for_model],
                **self._model_params,
            )
        except Exception as e:
            raise RuntimeError(f"Error in calling openai API: {e}") from e

        return self._parse_response(response)

    @tracer.llm(name="LLM Async Invoke")
    async def ainvoke(self, input: Union[str, List[Message], List[Dict[str, str]]]) -> ModelResponse:
        try:
            messages_for_model = self.input_to_message_list(input)
            response: ChatCompletion = await self.get_async_client().chat.completions.create(
                model=self.model,
                messages=[self._format_message(m) for m in messages_for_model],
                **self._model_params,
            )
        except Exception as e:
            raise RuntimeError(f"Error in calling openai API: {e}") from e

        return self._parse_response(response)

    @tracer.llm(name="LLM Stream")
    def stream(self, input: Union[str, List[Message], List[Dict[str, str]]]) -> Iterator[ModelResponse]:
        try:
            messages_for_model = self.input_to_message_list(input)
            completion = self.get_client().chat.completions.create(
                model=self.model,
                messages=[self._format_message(m) for m in messages_for_model],
                stream=True,
                stream_options={"include_usage": True},
                **self._model_params,
            )

            _full_tool_calls = []
            think_extractor = _ThinkTagStreamExtractor() if self.extract_think_tags else None

            for chunk in completion:
                response = ModelResponse()

                if chunk.choices:
                    delta: ChoiceDelta = chunk.choices[0].delta

                    reasoning_delta = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
                    if reasoning_delta:
                        response.parts.append(ThinkingContent(content=reasoning_delta))

                    content_delta = getattr(delta, "content", None)
                    if content_delta:
                        if think_extractor is not None:
                            for kind, segment in think_extractor.feed(content_delta):
                                if not segment:
                                    continue
                                if kind == "thinking":
                                    response.parts.append(ThinkingContent(content=segment))
                                else:
                                    response.parts.append(TextContent(content=segment))
                        else:
                            response.parts.append(TextContent(content=content_delta))

                    audio_part = _audio_delta_to_part(getattr(delta, "audio", None))
                    if audio_part is not None:
                        response.parts.append(audio_part)

                    if hasattr(delta, "tool_calls") and delta.tool_calls:
                        for tc in delta.tool_calls or []:
                            if _full_tool_calls and (not tc.id or tc.id == _full_tool_calls[-1].id):
                                if tc.function.name:
                                    _full_tool_calls[-1].name += tc.function.name
                                if tc.function.arguments:
                                    _full_tool_calls[-1].arguments += tc.function.arguments
                            else:
                                _full_tool_calls.append(
                                    ToolCall(id=tc.id, name=tc.function.name, arguments=tc.function.arguments)
                                )

                    if _full_tool_calls:
                        response.parts.extend(_full_tool_calls)

                    has_streamable = (
                        response.content
                        or response.thinking
                        or any(isinstance(p, AudioContent) for p in response.parts)
                    )
                    if has_streamable:
                        yield response
                    elif len(response.tool_calls) > 0 and chunk.choices[0].finish_reason == "tool_calls":
                        yield response

                else:
                    if hasattr(response, "usage"):
                        response.usage = self._get_usage(chunk)
                        yield response

        except Exception as e:
            raise RuntimeError(f"Error in calling openai API: {e}") from e

    @tracer.llm(name="LLM Async Astream")
    async def astream(self, input: Union[str, List[Message], List[Dict[str, str]]]) -> AsyncIterator[ModelResponse]:
        try:
            messages_for_model = self.input_to_message_list(input)
            completion = await self.get_async_client().chat.completions.create(
                model=self.model,
                messages=[self._format_message(m) for m in messages_for_model],
                stream=True,
                stream_options={"include_usage": True},
                **self._model_params,
            )

            _full_tool_calls = []
            think_extractor = _ThinkTagStreamExtractor() if self.extract_think_tags else None

            async for chunk in completion:
                response = ModelResponse()

                if chunk.choices:
                    delta: ChoiceDelta = chunk.choices[0].delta

                    reasoning_delta = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
                    if reasoning_delta:
                        response.parts.append(ThinkingContent(content=reasoning_delta))

                    content_delta = getattr(delta, "content", None)
                    if content_delta:
                        if think_extractor is not None:
                            for kind, segment in think_extractor.feed(content_delta):
                                if not segment:
                                    continue
                                if kind == "thinking":
                                    response.parts.append(ThinkingContent(content=segment))
                                else:
                                    response.parts.append(TextContent(content=segment))
                        else:
                            response.parts.append(TextContent(content=content_delta))

                    audio_part = _audio_delta_to_part(getattr(delta, "audio", None))
                    if audio_part is not None:
                        response.parts.append(audio_part)

                    if hasattr(delta, "tool_calls") and delta.tool_calls:
                        for tc in delta.tool_calls or []:
                            if _full_tool_calls and (not tc.id or tc.id == _full_tool_calls[-1].id):
                                if tc.function.name:
                                    _full_tool_calls[-1].name += tc.function.name
                                if tc.function.arguments:
                                    _full_tool_calls[-1].arguments += tc.function.arguments
                            else:
                                _full_tool_calls.append(
                                    ToolCall(id=tc.id, name=tc.function.name, arguments=tc.function.arguments)
                                )

                    if _full_tool_calls:
                        response.parts.extend(_full_tool_calls)

                    has_streamable = (
                        response.content
                        or response.thinking
                        or any(isinstance(p, AudioContent) for p in response.parts)
                    )
                    if has_streamable:
                        yield response
                    elif len(response.tool_calls) > 0 and chunk.choices[0].finish_reason == "tool_calls":
                        yield response

                else:
                    if hasattr(response, "usage"):
                        response.usage = self._get_usage(chunk)
                        yield response

        except Exception as e:
            raise RuntimeError(f"Error in calling openai API: {e}") from e

    # ---------------------------------------------------------------
    # Batch API
    # ---------------------------------------------------------------

    def _build_batch_line(
        self, custom_id: str, input: Union[str, List[Message], List[Dict[str, str]]]
    ) -> Dict[str, Any]:
        messages_for_model = self.input_to_message_list(input)
        body = {
            "model": self.model,
            "messages": [self._format_message(m) for m in messages_for_model],
            **self._model_params,
        }
        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }

    def _build_batch_file(
        self,
        inputs: List[Union[str, List[Message], List[Dict[str, str]]]],
        custom_ids: Optional[List[str]],
    ) -> Tuple[str, bytes]:
        if not inputs:
            raise ValueError("inputs must be a non-empty list")

        if custom_ids is None:
            custom_ids = [str(i) for i in range(len(inputs))]
        elif len(custom_ids) != len(inputs):
            raise ValueError("custom_ids must have the same length as inputs")
        elif len(set(custom_ids)) != len(custom_ids):
            raise ValueError("custom_ids must be unique")

        lines = [self._build_batch_line(cid, inp) for cid, inp in zip(custom_ids, inputs, strict=True)]
        jsonl = "\n".join(json.dumps(line) for line in lines).encode("utf-8")
        filename = f"batch_{uuid.uuid4().hex}.jsonl"
        return filename, jsonl

    def _parse_batch_output(self, content: str) -> Dict[str, BatchResultItem]:
        results: Dict[str, BatchResultItem] = {}
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            custom_id = record["custom_id"]
            if record.get("error"):
                results[custom_id] = BatchResultItem(custom_id=custom_id, error=record["error"])
                continue
            body = (record.get("response") or {}).get("body")
            completion = ChatCompletion.model_validate(body)
            results[custom_id] = BatchResultItem(custom_id=custom_id, response=self._parse_response(completion))
        return results

    def _parse_batch_error_file(self, content: str) -> Dict[str, BatchResultItem]:
        results: Dict[str, BatchResultItem] = {}
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            custom_id = record.get("custom_id")
            if custom_id is not None:
                results[custom_id] = BatchResultItem(custom_id=custom_id, error=record.get("error") or record)
        return results

    def create_batch(
        self,
        inputs: List[Union[str, List[Message], List[Dict[str, str]]]],
        custom_ids: Optional[List[str]] = None,
        completion_window: str = "24h",
        metadata: Optional[Dict[str, str]] = None,
    ) -> Batch:
        """Upload `inputs` (one per chat completion request) and submit them as a batch job.

        `custom_ids` correlates each input to its result in `get_batch_results()`;
        defaults to the input's index as a string. Returns the created `Batch` —
        use `batch.id` with `retrieve_batch()`/`poll_batch()`.
        """
        filename, jsonl = self._build_batch_file(inputs, custom_ids)
        client = self.get_client()
        input_file = client.files.create(file=(filename, jsonl), purpose="batch")
        return client.batches.create(
            input_file_id=input_file.id,
            endpoint="/v1/chat/completions",
            completion_window=completion_window,
            metadata=metadata,
        )

    async def acreate_batch(
        self,
        inputs: List[Union[str, List[Message], List[Dict[str, str]]]],
        custom_ids: Optional[List[str]] = None,
        completion_window: str = "24h",
        metadata: Optional[Dict[str, str]] = None,
    ) -> Batch:
        filename, jsonl = self._build_batch_file(inputs, custom_ids)
        client = self.get_async_client()
        input_file = await client.files.create(file=(filename, jsonl), purpose="batch")
        return await client.batches.create(
            input_file_id=input_file.id,
            endpoint="/v1/chat/completions",
            completion_window=completion_window,
            metadata=metadata,
        )

    def retrieve_batch(self, batch_id: str) -> Batch:
        return self.get_client().batches.retrieve(batch_id)

    async def aretrieve_batch(self, batch_id: str) -> Batch:
        return await self.get_async_client().batches.retrieve(batch_id)

    def cancel_batch(self, batch_id: str) -> Batch:
        return self.get_client().batches.cancel(batch_id)

    async def acancel_batch(self, batch_id: str) -> Batch:
        return await self.get_async_client().batches.cancel(batch_id)

    def _batch_validation_errors(self, batch: Batch) -> Dict[str, BatchResultItem]:
        """Top-level `batch.errors`.

        Populated on `status == "failed"`, when the input file itself was
        rejected before any request ran (no output/error file).
        """
        results: Dict[str, BatchResultItem] = {}
        if not batch.errors or not batch.errors.data:
            return results
        for i, err in enumerate(batch.errors.data):
            key = f"line:{err.line}" if err.line is not None else f"error:{i}"
            results[key] = BatchResultItem(
                custom_id=key,
                error={"code": err.code, "message": err.message, "param": err.param, "line": err.line},
            )
        return results

    def poll_batch(self, batch_id: str, poll_interval: float = 10.0, timeout: Optional[float] = None) -> Batch:
        """Block until the batch reaches a terminal status: completed, failed, expired or cancelled.

        Raises `TimeoutError` if `timeout` (seconds) elapses first.
        """
        start = time.monotonic()
        while True:
            batch = self.retrieve_batch(batch_id)
            if batch.status in _BATCH_TERMINAL_STATUSES:
                return batch
            if timeout is not None and time.monotonic() - start >= timeout:
                raise TimeoutError(f"Batch {batch_id} did not complete within {timeout}s (status={batch.status})")
            time.sleep(poll_interval)

    async def apoll_batch(self, batch_id: str, poll_interval: float = 10.0, timeout: Optional[float] = None) -> Batch:
        start = time.monotonic()
        while True:
            batch = await self.aretrieve_batch(batch_id)
            if batch.status in _BATCH_TERMINAL_STATUSES:
                return batch
            if timeout is not None and time.monotonic() - start >= timeout:
                raise TimeoutError(f"Batch {batch_id} did not complete within {timeout}s (status={batch.status})")
            await asyncio.sleep(poll_interval)

    def get_batch_results(self, batch_id: str) -> Dict[str, BatchResultItem]:
        """Fetch and parse a finished batch's results, keyed by `custom_id`.

        Covers every terminal status: `completed` results come from the output
        file; `expired`/`cancelled` batches may still carry partial output/error
        files for whatever ran before the cutoff; `failed` batches (input file
        rejected outright) surface `batch.errors` instead since no request ever ran.
        Raises `RuntimeError` if the batch hasn't reached a terminal status yet —
        call `poll_batch()`/`retrieve_batch()` first.
        """
        client = self.get_client()
        batch = self.retrieve_batch(batch_id)
        if batch.status not in _BATCH_TERMINAL_STATUSES:
            raise RuntimeError(f"Batch {batch_id} is not finished yet (status={batch.status})")

        results: Dict[str, BatchResultItem] = {}
        if batch.output_file_id:
            content = client.files.content(batch.output_file_id).text
            results.update(self._parse_batch_output(content))
        if batch.error_file_id:
            content = client.files.content(batch.error_file_id).text
            results.update(self._parse_batch_error_file(content))
        if not batch.output_file_id and not batch.error_file_id:
            results.update(self._batch_validation_errors(batch))
        return results

    async def aget_batch_results(self, batch_id: str) -> Dict[str, BatchResultItem]:
        client = self.get_async_client()
        batch = await self.aretrieve_batch(batch_id)
        if batch.status not in _BATCH_TERMINAL_STATUSES:
            raise RuntimeError(f"Batch {batch_id} is not finished yet (status={batch.status})")

        results: Dict[str, BatchResultItem] = {}
        if batch.output_file_id:
            content = (await client.files.content(batch.output_file_id)).text
            results.update(self._parse_batch_output(content))
        if batch.error_file_id:
            content = (await client.files.content(batch.error_file_id)).text
            results.update(self._parse_batch_error_file(content))
        if not batch.output_file_id and not batch.error_file_id:
            results.update(self._batch_validation_errors(batch))
        return results
