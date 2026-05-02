import functools
import inspect
from contextlib import contextmanager
from typing import Any


@functools.lru_cache(maxsize=None)
def _get_semconv() -> tuple:
    """Load openinference semantic conventions once and cache the result."""
    try:
        from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
        return OpenInferenceSpanKindValues, SpanAttributes
    except ImportError:
        return None, None


def _update_stream_state(
    chunk: Any,
    accumulated_content: str,
    latest_tool_calls: Any,
) -> tuple[str, Any]:
    if hasattr(chunk, "content") and chunk.content:
        accumulated_content += chunk.content
    if hasattr(chunk, "tool_calls") and chunk.tool_calls:
        latest_tool_calls = chunk.tool_calls
    return accumulated_content, latest_tool_calls


class _NoOpSpan:
    """Returned when telemetry is not initialized — all operations are no-ops."""

    def is_recording(self) -> bool:
        return False

    def set_attribute(self, *a, **kw) -> None:
        pass

    def set_status(self, *a, **kw) -> None:
        pass

    def record_exception(self, *a, **kw) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a) -> bool:
        return False


class DecoratorTracer:
    def __init__(self, tracer_name: str = "gwenflow"):
        self.tracer_name = tracer_name
        self._tracer = None

    def _ensure_tracer(self):
        if self._tracer is None:
            try:
                from opentelemetry import trace
                self._tracer = trace.get_tracer(self.tracer_name)
            except ImportError:
                pass
        return self._tracer

    def _resolve(self, name_attr: str, kind_name: str, name_override: str | None, instance: Any) -> tuple:
        kind_values, _ = _get_semconv()
        kind_val = getattr(kind_values, kind_name, None) if kind_values else None
        name = name_override or f"{kind_name}:{getattr(instance, name_attr, 'unknown')}"
        return name, kind_val

    @contextmanager
    def _start_span(self, name, kind, instance, func, args, kwargs):
        otel_tracer = self._ensure_tracer()
        _, SpanAttributes = _get_semconv()

        if otel_tracer is None:
            yield _NoOpSpan()
            return

        try:
            from opentelemetry.trace import StatusCode
            from .utils import prepare_llm_attributes
            from .utils import extract_user_inputs
        except ImportError:
            yield _NoOpSpan()
            return

        with otel_tracer.start_as_current_span(name) as span:
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

    def _finalize_span_output(self, span, content: str, tool_calls_json: str | None, fallback_obj: Any) -> None:
        if not span.is_recording():
            return
        _, SpanAttributes = _get_semconv()
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

    def _finalize_after(self, span, kind_name: str, result: Any, content: str, tool_calls: Any) -> None:
        if not span.is_recording():
            return
        try:
            from .utils import capture_llm_usage, capture_tool_calls
        except ImportError:
            return

        if kind_name == "LLM":
            capture_llm_usage(span, result)
        tc_json = capture_tool_calls(span, tool_calls)
        self._finalize_span_output(span, content or "", tc_json, result)

    def _wrap_logic(self, name_attr: str, kind_name: str, name_override: str | None = None):
        def decorator(func):

            # 1. ASYNC GENERATOR
            if inspect.isasyncgenfunction(func):
                @functools.wraps(func)
                async def wrapper(instance, *args, **kwargs):
                    name, kind_val = self._resolve(name_attr, kind_name, name_override, instance)
                    with self._start_span(name, kind_val, instance, func, args, kwargs) as span:
                        accumulated_content, latest_tool_calls, last_chunk = "", None, None
                        try:
                            async for chunk in func(instance, *args, **kwargs):
                                last_chunk = chunk
                                accumulated_content, latest_tool_calls = _update_stream_state(chunk, accumulated_content, latest_tool_calls)
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
                    name, kind_val = self._resolve(name_attr, kind_name, name_override, instance)
                    with self._start_span(name, kind_val, instance, func, args, kwargs) as span:
                        result = await func(instance, *args, **kwargs)
                        self._finalize_after(span, kind_name, result, getattr(result, "content", "") or "", getattr(result, "tool_calls", None))
                        return result
                return wrapper

            # 3. SYNC GENERATOR
            elif inspect.isgeneratorfunction(func):
                @functools.wraps(func)
                def wrapper(instance, *args, **kwargs):
                    name, kind_val = self._resolve(name_attr, kind_name, name_override, instance)
                    with self._start_span(name, kind_val, instance, func, args, kwargs) as span:
                        accumulated_content, latest_tool_calls, last_chunk = "", None, None
                        try:
                            for chunk in func(instance, *args, **kwargs):
                                last_chunk = chunk
                                accumulated_content, latest_tool_calls = _update_stream_state(chunk, accumulated_content, latest_tool_calls)
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
                    name, kind_val = self._resolve(name_attr, kind_name, name_override, instance)
                    with self._start_span(name, kind_val, instance, func, args, kwargs) as span:
                        result = func(instance, *args, **kwargs)
                        self._finalize_after(span, kind_name, result, getattr(result, "content", "") or "", getattr(result, "tool_calls", None))
                        return result
                return wrapper

        return decorator

    def llm(self, name: str | None = None):
        return self._wrap_logic("model", "LLM", name)

    def agent(self, name: str | None = None):
        return self._wrap_logic("name", "AGENT", name)

    def tool(self, name: str | None = None):
        return self._wrap_logic("name", "TOOL", name)

    def flow(self, name: str | None = None):
        return self._wrap_logic("name", "CHAIN", name)


tracer = DecoratorTracer()
