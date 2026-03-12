import json
from typing import Any

from openinference.semconv.trace import SpanAttributes
from opentelemetry.trace import Span


def prepare_llm_attributes(span: Span, instance: Any):
    if hasattr(instance, "model"):
        span.set_attribute(SpanAttributes.LLM_MODEL_NAME, str(instance.model))
    if hasattr(instance, "_model_params"):
        span.set_attribute(SpanAttributes.LLM_INVOCATION_PARAMETERS, json.dumps(instance._model_params, default=str))


def capture_llm_usage(span: Span, result: Any):
    if not hasattr(result, "usage") or not result.usage:
        return

    usage = result.usage
    # TODO change later to add more details in the token comsuption
    if getattr(usage, "input_tokens", None) is not None:
        span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_PROMPT, usage.input_tokens)

    if getattr(usage, "output_tokens", None) is not None:
        span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, usage.output_tokens)

    if getattr(usage, "total_tokens", None) is not None:
        span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_TOTAL, usage.total_tokens)


def capture_tool_calls(span: Span, tool_calls: Any):
    if not tool_calls:
        return
    try:
        tool_calls_dump = [tc.model_dump() for tc in tool_calls]
        tool_calls_json = json.dumps(tool_calls_dump, default=str)
        span.set_attribute("llm.tool_calls", tool_calls_json)

        return tool_calls_json
    except Exception:
        pass
    return None


def capture_impacts(span: Span, result: Any):
    if not hasattr(result, "impacts") or not result.impacts:
        return

    try:
        if hasattr(result.impacts, "model_dump_json"):
            impacts_json = result.impacts.model_dump_json()
        elif hasattr(result.impacts, "model_dump"):
            impacts_json = json.dumps(result.impacts.model_dump())
        elif hasattr(result.impacts, "dict"):
            impacts_json = json.dumps(result.impacts.dict())
        else:
            impacts_json = json.dumps(result.impacts)

        span.set_attribute("ecologits.impacts", impacts_json)
    except Exception:
        pass
