import dataclasses
import inspect
import json
from typing import Any, Callable


def safe_serialize(obj: Any) -> str:
    try:
        if hasattr(obj, "model_dump_json") and callable(obj.model_dump_json):
            return obj.model_dump_json()
        if dataclasses.is_dataclass(obj):
            return json.dumps(dataclasses.asdict(obj), default=str)
        return str(obj)
    except Exception:
        return str(obj)


def extract_user_inputs(func: Callable, args: tuple, kwargs: dict) -> str:
    try:
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        arg_dict = dict(bound_args.arguments)
        arg_dict.pop("self", None)
        arg_dict.pop("cls", None)

        for key in ("input", "query", "content", "tool_call"):
            if key in arg_dict:
                return safe_serialize(arg_dict[key])

        return json.dumps({k: str(v) for k, v in arg_dict.items()}, default=str)
    except Exception:
        return "Error capturing inputs"


def prepare_llm_attributes(span, instance: Any) -> None:
    try:
        from openinference.semconv.trace import SpanAttributes
    except ImportError:
        return

    if hasattr(instance, "model"):
        span.set_attribute(SpanAttributes.LLM_MODEL_NAME, str(instance.model))
    if hasattr(instance, "_model_params"):
        span.set_attribute(SpanAttributes.LLM_INVOCATION_PARAMETERS, json.dumps(instance._model_params, default=str))


def capture_llm_usage(span, result: Any) -> None:
    try:
        from openinference.semconv.trace import SpanAttributes
    except ImportError:
        return

    if not hasattr(result, "usage") or not result.usage:
        return

    usage = result.usage
    if getattr(usage, "input_tokens", None) is not None:
        span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_PROMPT, usage.input_tokens)
    if getattr(usage, "output_tokens", None) is not None:
        span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, usage.output_tokens)
    if getattr(usage, "total_tokens", None) is not None:
        span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_TOTAL, usage.total_tokens)


def capture_tool_calls(span, tool_calls: Any) -> str | None:
    if not tool_calls:
        return None
    try:
        tool_calls_data = []
        for tc in tool_calls:
            if hasattr(tc, "model_dump"):
                tool_calls_data.append(tc.model_dump())
            elif dataclasses.is_dataclass(tc):
                tool_calls_data.append(dataclasses.asdict(tc))
            else:
                tool_calls_data.append(str(tc))
        tool_calls_json = json.dumps(tool_calls_data, default=str)
        span.set_attribute("llm.tool_calls", tool_calls_json)
        return tool_calls_json
    except Exception:
        return None
