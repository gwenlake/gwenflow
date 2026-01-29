import functools
import inspect
import json
from contextlib import contextmanager
from typing import Any

from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
from opentelemetry import trace
from opentelemetry.trace import StatusCode, Tracer


def safe_serialize(obj: Any) -> str:
    try:
        if hasattr(obj, "model_dump_json"):
            return obj.model_dump_json()
        if hasattr(obj, "dict"):
            return json.dumps(obj.dict(), default=str)
        return str(obj)
    except Exception:
        return str(obj)


class DecoratorTracer:
    def __init__(self):
        self.tracer = trace.get_tracer("gwenflow")

    def _get_input_value(self, func, args, kwargs) -> str:
        """Fonction to capture the user's input."""
        try:
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            arg_dict = dict(bound_args.arguments)

            keys_to_remove = [k for k in arg_dict.keys() if k.startswith("self")]
            for k in keys_to_remove:
                del arg_dict[k]

            if "input" in arg_dict:
                return safe_serialize(arg_dict["input"])
            if "query" in arg_dict:
                return safe_serialize(arg_dict["query"])

            return json.dumps({k: str(v) for k, v in arg_dict.items()}, default=str)
        except Exception:
            return "Error capturing inputs"

    @contextmanager
    def _start_span(self, name, kind, instance, func, args, kwargs):
        with self.tracer.start_as_current_span(name) as span:
            span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, kind.value)

            input_val = self._get_input_value(func, (instance, *args), kwargs)
            span.set_attribute(SpanAttributes.INPUT_VALUE, str(input_val))

            try:
                yield span
            except Exception as e:
                span.set_status(StatusCode.ERROR, str(e))
                span.record_exception(e)
                raise

    def _attempt_capture_usage(self, span, chunk):
        """Helper to extract usage from LLM events specifically."""
        try:
            if hasattr(chunk, "root"):
                event = chunk.root
                event_type = getattr(event, "type", "")

                if event_type in ["response.completed", "response.done"]:
                    response = getattr(event, "response", None)
                    if response:
                        self._capture_usage_attributes(span, response)
        except Exception:
            pass

    def _capture_usage_attributes(self, span, result):
        """Extracts token counts from a result object and sets span attributes."""
        if not hasattr(result, "usage") or not result.usage:
            return

        usage = result.usage

        input_tokens = getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", None)
        if input_tokens is not None:
            span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_PROMPT, int(input_tokens))

        output_tokens = getattr(usage, "output_tokens", None) or getattr(usage, "completion_tokens", None)
        if output_tokens is not None:
            span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, int(output_tokens))

        total_tokens = getattr(usage, "total_tokens", None)
        if total_tokens is not None:
            span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_TOTAL, int(total_tokens))

        input_details = getattr(usage, "input_tokens_details", None)
        if input_details:
            cached = getattr(input_details, "cached_tokens", None)
            if cached is not None:
                span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_INPUT, int(cached))

        output_details = getattr(usage, "output_tokens_details", None) or getattr(usage, "completion_tokens_details", None)
        if output_details:
            reasoning = getattr(output_details, "reasoning_tokens", None)
            if reasoning is not None:
                span.set_attribute(SpanAttributes.LLM_COST_COMPLETION_DETAILS_REASONING, int(reasoning))

    def _finalize_stream_span(self, span, last_chunk):
        """Sets the final output of the span based on the last chunk received."""
        if last_chunk is None:
            return

        try:
            obj_to_serialize = last_chunk

            if hasattr(last_chunk, "root"):
                root = last_chunk.root
                if hasattr(root, "response") and getattr(root, "type", "") in ["response.completed", "response.done"]:
                    obj_to_serialize = root.response

            span.set_attribute(SpanAttributes.OUTPUT_VALUE, safe_serialize(obj_to_serialize))
            span.set_status(StatusCode.OK)
        except Exception:
            span.set_status(StatusCode.OK)

    def _prepare_llm(self, span, instance):
        if hasattr(instance, "model"):
            span.set_attribute(SpanAttributes.LLM_MODEL_NAME, instance.model)
        if hasattr(instance, "_model_params"):
            span.set_attribute(SpanAttributes.LLM_INVOCATION_PARAMETERS, json.dumps(instance._model_params, default=str))

    def _wrap_logic(self, name_attr, kind, name_override=None):
        def decorator(func):
            # 1. ASYNC GENERATOR (Streaming Async)
            if inspect.isasyncgenfunction(func):
                @functools.wraps(func)
                async def wrapper(instance, *args, **kwargs):
                    name = name_override or f"{kind.value}:{getattr(instance, name_attr, 'unknown')}"
                    with self._start_span(name, kind, instance, func, args, kwargs) as span:
                        if kind == OpenInferenceSpanKindValues.LLM:
                            self._prepare_llm(span, instance)

                        last_chunk = None
                        try:
                            async for chunk in func(instance, *args, **kwargs):
                                last_chunk = chunk
                                # Try capture usage on-the-fly for LLMs
                                if kind == OpenInferenceSpanKindValues.LLM:
                                    self._attempt_capture_usage(span, chunk)
                                yield chunk

                            self._finalize_stream_span(span, last_chunk)

                        except Exception as e:
                            span.record_exception(e)
                            span.set_status(StatusCode.ERROR, str(e))
                            raise
                return wrapper

            # 2. ASYNC FUNCTION (ainvoke / arun)
            elif inspect.iscoroutinefunction(func):
                @functools.wraps(func)
                async def wrapper(instance, *args, **kwargs):
                    name = name_override or f"{kind.value}:{getattr(instance, name_attr, 'unknown')}"
                    with self._start_span(name, kind, instance, func, args, kwargs) as span:
                        if kind == OpenInferenceSpanKindValues.LLM:
                            self._prepare_llm(span, instance)
                        result = await func(instance, *args, **kwargs)
                        if kind == OpenInferenceSpanKindValues.LLM:
                            self._capture_usage_attributes(span, result)
                        span.set_attribute(SpanAttributes.OUTPUT_VALUE, safe_serialize(result))
                        span.set_status(StatusCode.OK)
                        return result
                return wrapper

            # 3. SYNC GENERATOR (Streaming Sync)
            elif inspect.isgeneratorfunction(func):
                @functools.wraps(func)
                def wrapper(instance, *args, **kwargs):
                    name = name_override or f"{kind.value}:{getattr(instance, name_attr, 'unknown')}"
                    with self._start_span(name, kind, instance, func, args, kwargs) as span:
                        if kind == OpenInferenceSpanKindValues.LLM:
                            self._prepare_llm(span, instance)

                        last_chunk = None
                        try:
                            for chunk in func(instance, *args, **kwargs):
                                last_chunk = chunk
                                if kind == OpenInferenceSpanKindValues.LLM:
                                    self._attempt_capture_usage(span, chunk)
                                yield chunk

                            self._finalize_stream_span(span, last_chunk)

                        except Exception as e:
                            span.record_exception(e)
                            span.set_status(StatusCode.ERROR, str(e))
                            raise
                return wrapper

            # 4. SYNC FUNCTION (invoke / run)
            else:
                @functools.wraps(func)
                def wrapper(instance, *args, **kwargs):
                    name = name_override or f"{kind.value}:{getattr(instance, name_attr, 'unknown')}"
                    with self._start_span(name, kind, instance, func, args, kwargs) as span:
                        if kind == OpenInferenceSpanKindValues.LLM:
                            self._prepare_llm(span, instance)
                        result = func(instance, *args, **kwargs)
                        if kind == OpenInferenceSpanKindValues.LLM:
                            self._capture_usage_attributes(span, result)
                        span.set_attribute(SpanAttributes.OUTPUT_VALUE, safe_serialize(result))
                        span.set_status(StatusCode.OK)
                        return result
                return wrapper
        return decorator

    def llm(self, name=None):
        return self._wrap_logic("model", OpenInferenceSpanKindValues.LLM, name)

    def agent(self, name=None):
        return self._wrap_logic("name", OpenInferenceSpanKindValues.AGENT, name)

    def tool(self, name=None):
        return self._wrap_logic("name", OpenInferenceSpanKindValues.TOOL, name)

    def flow(self, name=None):
        return self._wrap_logic("name", OpenInferenceSpanKindValues.CHAIN, name)

Tracer = DecoratorTracer()