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


def _prepare_llm_attributes(span, instance: Any) -> None:
    model = getattr(instance, "model", None)
    if model is not None:
        span.set_attribute(sc.LLM_MODEL_NAME, str(model))
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


def _set_int(span, key: str, value: Any) -> None:
    if isinstance(value, int) and value > 0:
        span.set_attribute(key, value)


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
}
