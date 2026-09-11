import contextvars
import functools
import inspect
import json
from contextlib import contextmanager
from typing import Any, Callable

from gwenflow.logger import logger
from gwenflow.telemetry import _semconv as sc
from gwenflow.telemetry._settings import is_tracing_enabled, should_report_llm_usage
from gwenflow.telemetry.utils import (
    capture_agent_usage,
    capture_finish_reason,
    capture_llm_usage,
    capture_tool_calls,
    record_documents,
    record_inputs,
    record_outputs,
)

_telemetry_context: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "gwenflow_telemetry_context", default=None
)

_UNSET = object()

_BAGGAGE_VALUE_MAX_CHARS = 1024


def attribute_value(value: Any) -> Any:
    if isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple)) and all(isinstance(v, (str, bool, int, float)) for v in value):
        return list(value)
    return json.dumps(value, default=str)


def _baggage_value(value: Any) -> str | None:
    if isinstance(value, str):
        encoded = value
    elif isinstance(value, (bool, int, float)):
        encoded = str(value)
    else:
        encoded = json.dumps(value, default=str)
    if not encoded or len(encoded) > _BAGGAGE_VALUE_MAX_CHARS:
        return None
    return encoded


def _attach_baggage(metadata: dict[str, Any] | None) -> Callable[[], None]:
    if not metadata:
        return lambda: None
    try:
        from opentelemetry import baggage
        from opentelemetry import context as otel_context
    except ImportError:
        return lambda: None

    ctx = otel_context.get_current()
    for key, value in metadata.items():
        encoded = _baggage_value(value)
        if encoded is None:
            logger.debug(f"Telemetry context '{key}' is not carried as baggage (empty or over the size limit).")
            continue
        ctx = baggage.set_baggage(key, encoded, context=ctx)
    token = otel_context.attach(ctx)

    def restore() -> None:
        if otel_context.get_current() is ctx:
            otel_context.detach(token)

    return restore


def _suppress_key() -> object:
    try:
        from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY

        return _SUPPRESS_INSTRUMENTATION_KEY
    except ImportError:
        return "suppress_instrumentation"


def _suppressed() -> bool:
    try:
        from opentelemetry import context as otel_context
    except ImportError:
        return False
    return bool(otel_context.get_value(_suppress_key()))


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
        restore_baggage = _attach_baggage(metadata)
        try:
            yield
        finally:
            _telemetry_context.set(prev)
            restore_baggage()

    @contextmanager
    def suppress(self):
        try:
            from opentelemetry import context as otel_context
        except ImportError:
            yield
            return
        ctx = otel_context.set_value(_suppress_key(), True)
        token = otel_context.attach(ctx)
        try:
            yield
        finally:
            if otel_context.get_current() is ctx:
                otel_context.detach(token)

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
            span.set_attribute(key, attribute_value(value))

    def _record_start(self, span, kind_name: str, instance: Any, func: Any, args: tuple, kwargs: dict) -> None:
        try:
            self._apply_context(span)
            record_inputs(span, kind_name, instance, func, args, kwargs)
        except Exception as e:
            logger.debug(f"Telemetry failed to record span inputs: {e}")

    def _record_finish(
        self, span, kind_name: str, result: Any, content: Any = _UNSET, tool_calls: Any = _UNSET
    ) -> None:
        """Record outputs without ever masking a call that already succeeded.

        Streaming callers pass the accumulated `content`/`tool_calls`; the others
        let this read them off the result.
        """
        try:
            if content is _UNSET:
                content = getattr(result, "content", "") or ""
            if tool_calls is _UNSET:
                tool_calls = getattr(result, "tool_calls", None)
            self._finalize(span, kind_name, result, content, tool_calls)
        except Exception as e:
            logger.debug(f"Telemetry failed to record span outputs: {e}")

    def _finalize(self, span, kind_name: str, result_for_usage: Any, content: str, tool_calls: Any) -> None:
        if kind_name == "LLM":
            if should_report_llm_usage():
                capture_llm_usage(span, result_for_usage)
            capture_finish_reason(span, result_for_usage)
        elif kind_name == "AGENT":
            capture_agent_usage(span, result_for_usage)
        elif kind_name == "RETRIEVER":
            record_documents(span, sc.RETRIEVAL_DOCUMENTS, result_for_usage)
            return
        elif kind_name == "RERANKER":
            record_documents(span, sc.RERANKING_OUTPUT_DOCUMENTS, result_for_usage)
            return
        elif kind_name == "EMBEDDING":
            return
        tc_json = capture_tool_calls(span, tool_calls)
        record_outputs(span, content, tc_json, result_for_usage)

    def _ok(self, span) -> None:
        from opentelemetry.trace import StatusCode

        status = getattr(span, "status", None)
        if status is not None and status.status_code is StatusCode.ERROR:
            return
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
                    if not is_tracing_enabled() or _suppressed():
                        async for chunk in func(instance, *args, **kwargs):
                            yield chunk
                        return

                    from opentelemetry import context as otel_context
                    from opentelemetry import trace

                    span = self._get_tracer().start_span(make_name(instance))
                    ctx = trace.set_span_in_context(span)
                    token = otel_context.attach(ctx)
                    try:
                        self._record_start(span, kind_name, instance, func, args, kwargs)
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
                        self._record_finish(span, kind_name, last, content, tool_calls)
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
                    if not is_tracing_enabled() or _suppressed():
                        return await func(instance, *args, **kwargs)

                    from opentelemetry import context as otel_context
                    from opentelemetry import trace

                    span = self._get_tracer().start_span(make_name(instance))
                    token = otel_context.attach(trace.set_span_in_context(span))
                    try:
                        self._record_start(span, kind_name, instance, func, args, kwargs)
                        result = await func(instance, *args, **kwargs)
                        self._record_finish(span, kind_name, result)
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
                    if not is_tracing_enabled() or _suppressed():
                        yield from func(instance, *args, **kwargs)
                        return

                    from opentelemetry import context as otel_context
                    from opentelemetry import trace

                    span = self._get_tracer().start_span(make_name(instance))
                    ctx = trace.set_span_in_context(span)
                    token = otel_context.attach(ctx)
                    try:
                        self._record_start(span, kind_name, instance, func, args, kwargs)
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
                        self._record_finish(span, kind_name, last, content, tool_calls)
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
                if not is_tracing_enabled() or _suppressed():
                    return func(instance, *args, **kwargs)

                from opentelemetry import context as otel_context
                from opentelemetry import trace

                span = self._get_tracer().start_span(make_name(instance))
                token = otel_context.attach(trace.set_span_in_context(span))
                try:
                    self._record_start(span, kind_name, instance, func, args, kwargs)
                    result = func(instance, *args, **kwargs)
                    self._record_finish(span, kind_name, result)
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

    def retriever(self, name: str | None = None):
        return self._wrap_logic("name", "RETRIEVER", name)

    def embedding(self, name: str | None = None):
        return self._wrap_logic("model", "EMBEDDING", name)

    def reranker(self, name: str | None = None):
        return self._wrap_logic("model", "RERANKER", name)


tracer = DecoratorTracer()
