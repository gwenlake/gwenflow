import contextvars
import functools
import inspect
from contextlib import contextmanager
from typing import Any

from gwenflow.telemetry import _semconv as sc
from gwenflow.telemetry._settings import is_tracing_enabled
from gwenflow.telemetry.utils import capture_llm_usage, capture_tool_calls, record_inputs, record_outputs

_telemetry_context: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "gwenflow_telemetry_context", default=None
)


def _update_stream_state(chunk: Any, content: str, tool_calls: Any) -> tuple[str, Any]:
    if getattr(chunk, "content", None):
        content += chunk.content
    if getattr(chunk, "tool_calls", None):
        tool_calls = chunk.tool_calls
    return content, tool_calls


class DecoratorTracer:
    def __init__(self, tracer_name: str = "gwenflow"):
        self.tracer_name = tracer_name
        self._tracer = None

    def _get_tracer(self):
        if self._tracer is None:
            from opentelemetry import trace

            self._tracer = trace.get_tracer(self.tracer_name)
        return self._tracer

    @contextmanager
    def context(self, metadata: dict[str, Any] | None = None):
        prev = _telemetry_context.get()
        merged: dict[str, Any] = dict(prev) if prev else {}
        if metadata:
            merged.update(metadata)
        _telemetry_context.set(merged)
        try:
            yield
        finally:
            _telemetry_context.set(prev)

    def session(
        self,
        session_id: str | None = None,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        attrs: dict[str, Any] = dict(metadata or {})
        if session_id is not None:
            attrs[sc.SESSION_ID] = str(session_id)
        if user_id is not None:
            attrs[sc.USER_ID] = str(user_id)
        return self.context(metadata=attrs or None)

    def _apply_context(self, span) -> None:
        attrs = _telemetry_context.get()
        if not attrs:
            return
        for key, value in attrs.items():
            span.set_attribute(key, value if isinstance(value, (str, bool, int, float)) else str(value))

    def _finalize(self, span, kind_name: str, result_for_usage: Any, content: str, tool_calls: Any) -> None:
        if kind_name == "LLM":
            capture_llm_usage(span, result_for_usage)
        tc_json = capture_tool_calls(span, tool_calls)
        record_outputs(span, content, tc_json, result_for_usage)

    def _ok(self, span) -> None:
        from opentelemetry.trace import StatusCode

        span.set_status(StatusCode.OK)

    def _error(self, span, exc: BaseException) -> None:
        from opentelemetry.trace import StatusCode

        span.set_status(StatusCode.ERROR, str(exc))
        span.record_exception(exc)

    def _wrap_logic(self, name_attr: str, kind_name: str, name_override: str | None = None):
        def decorator(func):
            def make_name(instance: Any) -> str:
                return name_override or f"{kind_name}:{getattr(instance, name_attr, 'unknown')}"

            # 1. ASYNC GENERATOR
            if inspect.isasyncgenfunction(func):

                @functools.wraps(func)
                async def wrapper(instance, *args, **kwargs):
                    if not is_tracing_enabled():
                        async for chunk in func(instance, *args, **kwargs):
                            yield chunk
                        return

                    from opentelemetry import context as otel_context
                    from opentelemetry import trace

                    span = self._get_tracer().start_span(make_name(instance))
                    ctx = trace.set_span_in_context(span)
                    token = otel_context.attach(ctx)
                    try:
                        self._apply_context(span)
                        record_inputs(span, kind_name, instance, func, args, kwargs)
                    finally:
                        otel_context.detach(token)

                    agen = func(instance, *args, **kwargs)
                    content, tool_calls, last = "", None, None
                    try:
                        while True:
                            token = otel_context.attach(ctx)
                            try:
                                chunk = await agen.__anext__()
                            except StopAsyncIteration:
                                break
                            finally:
                                otel_context.detach(token)
                            last = chunk
                            content, tool_calls = _update_stream_state(chunk, content, tool_calls)
                            yield chunk
                        self._finalize(span, kind_name, last, content, tool_calls)
                        self._ok(span)
                    except Exception as e:
                        self._error(span, e)
                        raise
                    finally:
                        span.end()

                return wrapper

            # 2. ASYNC FUNCTION
            if inspect.iscoroutinefunction(func):

                @functools.wraps(func)
                async def wrapper(instance, *args, **kwargs):
                    if not is_tracing_enabled():
                        return await func(instance, *args, **kwargs)

                    from opentelemetry import context as otel_context
                    from opentelemetry import trace

                    span = self._get_tracer().start_span(make_name(instance))
                    token = otel_context.attach(trace.set_span_in_context(span))
                    try:
                        self._apply_context(span)
                        record_inputs(span, kind_name, instance, func, args, kwargs)
                        result = await func(instance, *args, **kwargs)
                        self._finalize(
                            span,
                            kind_name,
                            result,
                            getattr(result, "content", "") or "",
                            getattr(result, "tool_calls", None),
                        )
                        self._ok(span)
                        return result
                    except Exception as e:
                        self._error(span, e)
                        raise
                    finally:
                        otel_context.detach(token)
                        span.end()

                return wrapper

            # 3. SYNC GENERATOR
            if inspect.isgeneratorfunction(func):

                @functools.wraps(func)
                def wrapper(instance, *args, **kwargs):
                    if not is_tracing_enabled():
                        yield from func(instance, *args, **kwargs)
                        return

                    from opentelemetry import context as otel_context
                    from opentelemetry import trace

                    span = self._get_tracer().start_span(make_name(instance))
                    ctx = trace.set_span_in_context(span)
                    token = otel_context.attach(ctx)
                    try:
                        self._apply_context(span)
                        record_inputs(span, kind_name, instance, func, args, kwargs)
                    finally:
                        otel_context.detach(token)

                    gen = func(instance, *args, **kwargs)
                    content, tool_calls, last = "", None, None
                    try:
                        while True:
                            token = otel_context.attach(ctx)
                            try:
                                chunk = next(gen)
                            except StopIteration:
                                break
                            finally:
                                otel_context.detach(token)
                            last = chunk
                            content, tool_calls = _update_stream_state(chunk, content, tool_calls)
                            yield chunk
                        self._finalize(span, kind_name, last, content, tool_calls)
                        self._ok(span)
                    except Exception as e:
                        self._error(span, e)
                        raise
                    finally:
                        span.end()

                return wrapper

            # 4. SYNC FUNCTION
            @functools.wraps(func)
            def wrapper(instance, *args, **kwargs):
                if not is_tracing_enabled():
                    return func(instance, *args, **kwargs)

                from opentelemetry import context as otel_context
                from opentelemetry import trace

                span = self._get_tracer().start_span(make_name(instance))
                token = otel_context.attach(trace.set_span_in_context(span))
                try:
                    self._apply_context(span)
                    record_inputs(span, kind_name, instance, func, args, kwargs)
                    result = func(instance, *args, **kwargs)
                    self._finalize(
                        span,
                        kind_name,
                        result,
                        getattr(result, "content", "") or "",
                        getattr(result, "tool_calls", None),
                    )
                    self._ok(span)
                    return result
                except Exception as e:
                    self._error(span, e)
                    raise
                finally:
                    otel_context.detach(token)
                    span.end()

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
