import dataclasses
import inspect
import json
from typing import Any, Callable

from gwenflow.telemetry import _semconv as sc
from gwenflow.telemetry._settings import (
    REDACTED,
    should_capture_inputs,
    should_capture_outputs,
    truncate,
)

_INPUT_KEYS = ("input", "query", "content", "tool_call", "messages", "prompt")


def safe_serialize(obj: Any) -> str:
    try:
        if obj is None:
            return ""
        if isinstance(obj, str):
            return obj
        if hasattr(obj, "model_dump_json") and callable(obj.model_dump_json):
            return obj.model_dump_json()
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return json.dumps(dataclasses.asdict(obj), default=str)
        return str(obj)
    except Exception:
        try:
            return str(obj)
        except Exception:
            return "<unserializable>"


def extract_user_inputs(func: Callable, args: tuple, kwargs: dict) -> str:
    try:
        sig = inspect.signature(func)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        arg_dict = dict(bound.arguments)
        arg_dict.pop("self", None)
        arg_dict.pop("cls", None)

        for key in _INPUT_KEYS:
            if key in arg_dict and arg_dict[key] is not None:
                return safe_serialize(arg_dict[key])

        return json.dumps({k: safe_serialize(v) for k, v in arg_dict.items()}, default=str)
    except Exception:
        return ""


def _find_tool_name(args: tuple, kwargs: dict) -> str | None:
    for value in (*args, *kwargs.values()):
        name = getattr(value, "name", None)
        if isinstance(name, str) and name:
            return name
    return None


def record_inputs(span, kind_name: str, instance: Any, func: Callable, args: tuple, kwargs: dict) -> None:
    span.set_attribute(sc.OPENINFERENCE_SPAN_KIND, _SPAN_KINDS[kind_name])

    if should_capture_inputs():
        span.set_attribute(sc.INPUT_VALUE, truncate(extract_user_inputs(func, (instance, *args), kwargs)))
    else:
        span.set_attribute(sc.INPUT_VALUE, REDACTED)

    if kind_name == "LLM":
        _prepare_llm_attributes(span, instance)
    elif kind_name == "AGENT":
        name = getattr(instance, "name", None)
        if name:
            span.set_attribute(sc.AGENT_NAME, str(name))
    elif kind_name == "TOOL":
        tool_name = _find_tool_name(args, kwargs)
        if tool_name:
            span.set_attribute(sc.TOOL_NAME, tool_name)
    elif kind_name == "EMBEDDING":
        model = getattr(instance, "model", None)
        if model:
            span.set_attribute(sc.EMBEDDING_MODEL_NAME, str(model))
    elif kind_name == "RERANKER":
        model = getattr(instance, "model", None)
        if model:
            span.set_attribute(sc.RERANKING_MODEL_NAME, str(model))
        top_k = getattr(instance, "top_k", None)
        if isinstance(top_k, int):
            span.set_attribute(sc.RERANKING_TOP_K, top_k)


_PROVIDER_BY_CLASS = {
    "ChatOpenAI": "openai",
    "ChatAzureOpenAI": "azure_openai",
    "ChatGwenlake": "gwenlake",
    "ChatDeepSeek": "deepseek",
    "ChatOllama": "ollama",
    "ChatAnthropic": "anthropic",
    "ChatMistral": "mistral",
    "ChatGoogle": "google",
}


def _resolve_provider(instance: Any) -> str | None:
    for klass in type(instance).__mro__:
        provider = _PROVIDER_BY_CLASS.get(klass.__name__)
        if provider:
            return provider
    return None


def _prepare_llm_attributes(span, instance: Any) -> None:
    model = getattr(instance, "model", None)
    if model is not None:
        span.set_attribute(sc.LLM_MODEL_NAME, str(model))
    provider = _resolve_provider(instance)
    if provider:
        span.set_attribute(sc.LLM_PROVIDER, provider)
    params = getattr(instance, "_model_params", None)
    if params:
        span.set_attribute(sc.LLM_INVOCATION_PARAMETERS, truncate(json.dumps(params, default=str)))


def capture_llm_usage(span, result: Any) -> None:
    usage = getattr(result, "usage", None)
    if not usage:
        return
    _set_int(span, sc.LLM_TOKEN_COUNT_PROMPT, getattr(usage, "input_tokens", None))
    _set_int(span, sc.LLM_TOKEN_COUNT_COMPLETION, getattr(usage, "output_tokens", None))
    _set_int(span, sc.LLM_TOKEN_COUNT_TOTAL, getattr(usage, "total_tokens", None))
    _set_int(span, sc.LLM_TOKEN_COUNT_PROMPT_CACHE_READ, getattr(usage, "cache_read_tokens", None))
    _set_int(span, sc.LLM_TOKEN_COUNT_PROMPT_CACHE_WRITE, getattr(usage, "cache_write_tokens", None))
    _set_int(span, sc.LLM_TOKEN_COUNT_PROMPT_AUDIO, getattr(usage, "input_audio_tokens", None))
    _set_int(span, sc.LLM_TOKEN_COUNT_COMPLETION_AUDIO, getattr(usage, "output_audio_tokens", None))
    details = getattr(usage, "details", None)
    if isinstance(details, dict):
        _set_int(span, sc.LLM_TOKEN_COUNT_COMPLETION_REASONING, details.get("reasoning_tokens"))


def capture_finish_reason(span, result: Any) -> None:
    finish_reason = getattr(result, "finish_reason", None)
    if finish_reason:
        span.set_attribute(sc.LLM_FINISH_REASON, str(finish_reason))


def capture_agent_usage(span, result: Any) -> None:
    usage = getattr(result, "usage", None)
    if not usage:
        return
    _set_int(span, sc.AGENT_LLM_REQUESTS, getattr(usage, "requests", None))
    _set_int(span, sc.AGENT_TOOL_CALLS, getattr(usage, "tool_calls", None))


def _set_int(span, key: str, value: Any) -> None:
    if isinstance(value, int) and value > 0:
        span.set_attribute(key, value)


def _document_fields(doc: Any) -> dict:
    if hasattr(doc, "to_dict") and callable(doc.to_dict):
        try:
            return doc.to_dict()
        except Exception:
            pass
    if isinstance(doc, dict):
        return doc
    return {"content": safe_serialize(doc)}


def record_documents(span, prefix: str, documents: Any) -> None:
    """Emit OpenInference document attributes for a retriever / reranker result.

    Content is redacted when output capture is off; id / score stay as they are
    non-sensitive routing metadata.
    """
    if not documents:
        return
    capture = should_capture_outputs()
    try:
        for i, doc in enumerate(documents):
            fields = _document_fields(doc)
            base = f"{prefix}.{i}.document"
            doc_id = fields.get("id")
            if doc_id is not None:
                span.set_attribute(f"{base}.id", str(doc_id))
            score = fields.get("score")
            if isinstance(score, (int, float)):
                span.set_attribute(f"{base}.score", float(score))
            content = fields.get("content")
            if content is not None:
                span.set_attribute(f"{base}.content", truncate(str(content)) if capture else REDACTED)
            metadata = fields.get("metadata")
            if capture and metadata:
                span.set_attribute(f"{base}.metadata", truncate(json.dumps(metadata, default=str)))
    except Exception:
        pass


def capture_tool_calls(span, tool_calls: Any) -> str | None:
    if not tool_calls:
        return None
    try:
        data = []
        for tc in tool_calls:
            if hasattr(tc, "model_dump") and callable(tc.model_dump):
                data.append(tc.model_dump())
            elif dataclasses.is_dataclass(tc) and not isinstance(tc, type):
                data.append(dataclasses.asdict(tc))
            else:
                data.append(str(tc))
        tool_calls_json = json.dumps(data, default=str)
        if should_capture_outputs():
            span.set_attribute(sc.LLM_TOOL_CALLS, truncate(tool_calls_json))
        return tool_calls_json
    except Exception:
        return None


def record_outputs(span, content: str | None, tool_calls_json: str | None, fallback: Any) -> None:
    if not should_capture_outputs():
        span.set_attribute(sc.OUTPUT_VALUE, REDACTED)
        return
    if content:
        span.set_attribute(sc.OUTPUT_VALUE, truncate(content))
    elif tool_calls_json:
        span.set_attribute(sc.OUTPUT_VALUE, truncate(f"Tool Calls: {tool_calls_json}"))
    elif fallback is not None:
        span.set_attribute(sc.OUTPUT_VALUE, truncate(safe_serialize(fallback)))


_SPAN_KINDS = {
    "LLM": sc.SPAN_KIND_LLM,
    "AGENT": sc.SPAN_KIND_AGENT,
    "TOOL": sc.SPAN_KIND_TOOL,
    "CHAIN": sc.SPAN_KIND_CHAIN,
    "RETRIEVER": sc.SPAN_KIND_RETRIEVER,
    "EMBEDDING": sc.SPAN_KIND_EMBEDDING,
    "RERANKER": sc.SPAN_KIND_RERANKER,
}
