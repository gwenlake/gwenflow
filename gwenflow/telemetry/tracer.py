import functools
import inspect
from contextlib import contextmanager
from typing import Any

from opentelemetry import baggage


def _get_span_kind_values():
    try:
        from openinference.semconv.trace import OpenInferenceSpanKindValues

        return OpenInferenceSpanKindValues
    except ImportError:
        return None


def _get_span_attributes():
    try:
        from openinference.semconv.trace import SpanAttributes

        return SpanAttributes
    except ImportError:
        return None


def _get_tracer(tracer_name="gwenflow"):
    try:
        from opentelemetry import trace

        return trace.get_tracer(tracer_name)
    except ImportError:
        return None


class _NoOpSpan:
    """NoOp when telemetry is not initiate."""

    def is_recording(self):
        return False

    def set_attribute(self, *a, **kw):
        pass

    def set_status(self, *a, **kw):
        pass

    def record_exception(self, *a, **kw):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass


class DecoratorTracer:
    def __init__(self, tracer_name="gwenflow"):
        self.tracer_name = tracer_name
        self._tracer = None

    def _ensure_tracer(self):
        if self._tracer is None:
            self._tracer = _get_tracer(self.tracer_name)
        return self._tracer

    @contextmanager
    def _start_span(self, name, kind, instance, func, args, kwargs):
        otel_tracer = self._ensure_tracer()
        SpanAttributes = _get_span_attributes()  # noqa: N806

        if otel_tracer is None:
            yield _NoOpSpan()
            return

        try:
            from opentelemetry.trace import StatusCode

            from .llm_utils import prepare_llm_attributes
            from .utils import extract_user_inputs
        except ImportError:
            yield _NoOpSpan()
            return

        with otel_tracer.start_as_current_span(name) as span:
            session_id = baggage.get_baggage("gwenflow.session_id")
            if session_id and span.is_recording():
                span.set_attribute("gwenflow.session_id", session_id)
            if span.is_recording() and SpanAttributes and kind:
                span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, kind.value)
                input_val = extract_user_inputs(func, (instance, *args), kwargs)
                span.set_attribute(SpanAttributes.INPUT_VALUE, str(input_val))

                if hasattr(kind, "value") and kind.value == "LLM":
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

        SpanAttributes = _get_span_attributes()  # noqa: N806
        if not SpanAttributes:
            return

        try:
            from opentelemetry.trace import StatusCode

            from .utils import safe_serialize
        except ImportError:
            return

        if content:
            span.set_attribute(SpanAttributes.OUTPUT_VALUE, content)
        elif tool_calls_json:
            span.set_attribute(SpanAttributes.OUTPUT_VALUE, f"Tool Calls: {tool_calls_json}")
        elif fallback_obj is not None:
            span.set_attribute(SpanAttributes.OUTPUT_VALUE, safe_serialize(fallback_obj))

        span.set_status(StatusCode.OK)

    def _wrap_logic(self, name_attr, kind_name, name_override=None):
        def decorator(func):
            # 1. ASYNC GENERATOR
            if inspect.isasyncgenfunction(func):

                @functools.wraps(func)
                async def wrapper(instance, *args, **kwargs):
                    kind = _get_span_kind_values()
                    kind_val = getattr(kind, kind_name, None) if kind else None
                    name = name_override or f"{kind_name}:{getattr(instance, name_attr, 'unknown')}"
                    with self._start_span(name, kind_val, instance, func, args, kwargs) as span:
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
                            self._finalize_after(span, kind_name, last_chunk, accumulated_content, latest_tool_calls)
                        except Exception as e:
                            if span.is_recording():
                                span.record_exception(e)
                            raise

                return wrapper

            # 2. ASYNC FUNCTION
            elif inspect.iscoroutinefunction(func):

                @functools.wraps(func)
                async def wrapper(instance, *args, **kwargs):
                    kind = _get_span_kind_values()
                    kind_val = getattr(kind, kind_name, None) if kind else None
                    name = name_override or f"{kind_name}:{getattr(instance, name_attr, 'unknown')}"
                    with self._start_span(name, kind_val, instance, func, args, kwargs) as span:
                        result = await func(instance, *args, **kwargs)
                        self._finalize_after(
                            span, kind_name, result, getattr(result, "content", ""), getattr(result, "tool_calls", None)
                        )
                        return result

                return wrapper

            # 3. SYNC GENERATOR
            elif inspect.isgeneratorfunction(func):

                @functools.wraps(func)
                def wrapper(instance, *args, **kwargs):
                    kind = _get_span_kind_values()
                    kind_val = getattr(kind, kind_name, None) if kind else None
                    name = name_override or f"{kind_name}:{getattr(instance, name_attr, 'unknown')}"
                    with self._start_span(name, kind_val, instance, func, args, kwargs) as span:
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
                            self._finalize_after(span, kind_name, last_chunk, accumulated_content, latest_tool_calls)
                        except Exception as e:
                            if span.is_recording():
                                span.record_exception(e)
                            raise

                return wrapper

            # 4. SYNC FUNCTION
            else:

                @functools.wraps(func)
                def wrapper(instance, *args, **kwargs):
                    kind = _get_span_kind_values()
                    kind_val = getattr(kind, kind_name, None) if kind else None
                    name = name_override or f"{kind_name}:{getattr(instance, name_attr, 'unknown')}"
                    with self._start_span(name, kind_val, instance, func, args, kwargs) as span:
                        result = func(instance, *args, **kwargs)
                        self._finalize_after(
                            span, kind_name, result, getattr(result, "content", ""), getattr(result, "tool_calls", None)
                        )
                        return result

                return wrapper

        return decorator

    def _finalize_after(self, span, kind_name, result, content, tool_calls):
        if not span.is_recording():
            return
        try:
            from .llm_utils import capture_llm_usage, capture_tool_calls
        except ImportError:
            return

        tc_json = None
        if kind_name == "LLM":
            capture_llm_usage(span, result)
            tc_json = capture_tool_calls(span, tool_calls)
        else:
            tc_json = capture_tool_calls(span, tool_calls)

        self._finalize_span_output(span, content or "", tc_json, result)

    def llm(self, name=None):
        return self._wrap_logic("model", "LLM", name)

    def agent(self, name=None):
        return self._wrap_logic("name", "AGENT", name)

    def tool(self, name=None):
        return self._wrap_logic("name", "TOOL", name)

    def flow(self, name=None):
        return self._wrap_logic("name", "CHAIN", name)


tracer = DecoratorTracer()
