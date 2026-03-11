import functools
import inspect
from contextlib import contextmanager
from typing import Any

from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
from opentelemetry import trace
from opentelemetry.trace import StatusCode

from .llm_utils import capture_llm_usage, capture_tool_calls, prepare_llm_attributes
from .utils import extract_user_inputs, safe_serialize


class DecoratorTracer:
    def __init__(self, tracer_name="gwenflow"):
        self.tracer = trace.get_tracer(tracer_name)

    @contextmanager
    def _start_span(self, name, kind, instance, func, args, kwargs):
        with self.tracer.start_as_current_span(name) as span:
            if span.is_recording():
                span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, kind.value)
                input_val = extract_user_inputs(func, (instance, *args), kwargs)
                span.set_attribute(SpanAttributes.INPUT_VALUE, str(input_val))

                if kind == OpenInferenceSpanKindValues.LLM:
                    prepare_llm_attributes(span, instance)

            try:
                yield span
            except Exception as e:
                if span.is_recording():
                    span.set_status(StatusCode.ERROR, str(e))
                    span.record_exception(e)
                raise

    def _finalize_span_output(self, span, content: str, tool_calls_json: str, fallback_obj: Any):
        if not span.is_recording():
            return

        if content:
            span.set_attribute(SpanAttributes.OUTPUT_VALUE, content)
        elif tool_calls_json:
            span.set_attribute(SpanAttributes.OUTPUT_VALUE, f"Tool Calls: {tool_calls_json}")
        elif fallback_obj is not None:
            span.set_attribute(SpanAttributes.OUTPUT_VALUE, safe_serialize(fallback_obj))

        span.set_status(StatusCode.OK)

    def _wrap_logic(self, name_attr, kind, name_override=None):
        def decorator(func):
            # 1. ASYNC GENERATOR (Streaming Async)
            if inspect.isasyncgenfunction(func):

                @functools.wraps(func)
                async def wrapper(instance, *args, **kwargs):
                    name = name_override or f"{kind.value}:{getattr(instance, name_attr, 'unknown')}"
                    with self._start_span(name, kind, instance, func, args, kwargs) as span:
                        last_chunk = None
                        accumulated_content = ""
                        latest_tool_calls = None

                        try:
                            async for chunk in func(instance, *args, **kwargs):
                                last_chunk = chunk

                                if hasattr(chunk, "content") and chunk.content:
                                    accumulated_content += chunk.content

                                if hasattr(chunk, "tool_calls") and chunk.tool_calls:
                                    latest_tool_calls = chunk.tool_calls

                                yield chunk

                            if kind == OpenInferenceSpanKindValues.LLM:
                                capture_llm_usage(span, last_chunk)
                            tc_json = capture_tool_calls(span, latest_tool_calls)
                            self._finalize_span_output(span, accumulated_content, tc_json, last_chunk)

                        except Exception as e:
                            if span.is_recording():
                                span.record_exception(e)
                                span.set_status(StatusCode.ERROR, str(e))
                            raise

                return wrapper

            # 2. ASYNC FUNCTION (ainvoke)
            elif inspect.iscoroutinefunction(func):

                @functools.wraps(func)
                async def wrapper(instance, *args, **kwargs):
                    name = name_override or f"{kind.value}:{getattr(instance, name_attr, 'unknown')}"
                    with self._start_span(name, kind, instance, func, args, kwargs) as span:
                        result = await func(instance, *args, **kwargs)

                        content = getattr(result, "content", "")
                        tc_json = None

                        if kind == OpenInferenceSpanKindValues.LLM:
                            capture_llm_usage(span, result)
                            tc_json = capture_tool_calls(span, getattr(result, "tool_calls", None))

                        self._finalize_span_output(span, content, tc_json, result)
                        return result

                return wrapper

            # 3. SYNC GENERATOR (Streaming Sync)
            elif inspect.isgeneratorfunction(func):

                @functools.wraps(func)
                def wrapper(instance, *args, **kwargs):
                    name = name_override or f"{kind.value}:{getattr(instance, name_attr, 'unknown')}"
                    with self._start_span(name, kind, instance, func, args, kwargs) as span:
                        last_chunk = None
                        accumulated_content = ""
                        latest_tool_calls = None

                        try:
                            for chunk in func(instance, *args, **kwargs):
                                last_chunk = chunk
                                if hasattr(chunk, "content") and chunk.content:
                                    accumulated_content += chunk.content
                                if hasattr(chunk, "tool_calls") and chunk.tool_calls:
                                    latest_tool_calls = chunk.tool_calls
                                yield chunk

                            if kind == OpenInferenceSpanKindValues.LLM:
                                capture_llm_usage(span, last_chunk)
                            tc_json = capture_tool_calls(span, latest_tool_calls)
                            self._finalize_span_output(span, accumulated_content, tc_json, last_chunk)

                        except Exception as e:
                            if span.is_recording():
                                span.record_exception(e)
                                span.set_status(StatusCode.ERROR, str(e))
                            raise

                return wrapper

            # 4. SYNC FUNCTION (invoke)
            else:

                @functools.wraps(func)
                def wrapper(instance, *args, **kwargs):
                    name = name_override or f"{kind.value}:{getattr(instance, name_attr, 'unknown')}"
                    with self._start_span(name, kind, instance, func, args, kwargs) as span:
                        result = func(instance, *args, **kwargs)

                        content = getattr(result, "content", "")
                        tc_json = None

                        if kind == OpenInferenceSpanKindValues.LLM:
                            capture_llm_usage(span, result)
                            tc_json = capture_tool_calls(span, getattr(result, "tool_calls", None))

                        self._finalize_span_output(span, content, tc_json, result)
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


tracer = DecoratorTracer()
