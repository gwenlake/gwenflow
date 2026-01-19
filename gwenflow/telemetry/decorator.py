import functools
import json

from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
from opentelemetry import trace
from opentelemetry.trace import StatusCode


class DecoratorTracer:
    def __init__(self, tracer_name: str = "gwenflow"):
        self.tracer = trace.get_tracer(tracer_name)

    def _is_enabled(self) -> bool:
        provider = trace.get_tracer_provider()
        # Vérifie si un fournisseur de traces est configuré et actif
        return hasattr(provider, "resource") and provider.resource.attributes.get("service.name") != "unknown_service"

    def _inject_topology(self, span, self_agent):
        if hasattr(self_agent, "parent_flow_id") and self_agent.parent_flow_id:
            span.set_attribute("gwenflow.topology.parent_id", self_agent.parent_flow_id)

        if hasattr(self_agent, "depends_on") and self_agent.depends_on:
            span.set_attribute("gwenflow.topology.depends_on", self_agent.depends_on)

    def agent(self, name: str = None):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(self_agent, *args, **kwargs):
                if not self._is_enabled():
                    return func(self_agent, *args, **kwargs)

                session_id = kwargs.get("session_id") or getattr(self_agent, "session_id", "no_session")
                span_name = name or f"Agent:{self_agent.name}"

                with self.tracer.start_as_current_span(span_name) as span:
                    span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.AGENT.value)
                    span.set_attribute(SpanAttributes.SESSION_ID, session_id)
                    span.set_attribute("agent.id", str(getattr(self_agent, "id", "")))
                    span.set_attribute("agent.name", self_agent.name)

                    self._inject_topology(span, self_agent)

                    query = kwargs.get("query") or (args[0] if args else "None")
                    span.set_attribute(SpanAttributes.INPUT_VALUE, str(query))

                    try:
                        result = func(self_agent, *args, **kwargs)

                        span.set_attribute(SpanAttributes.OUTPUT_VALUE, str(result))
                        span.set_status(StatusCode.OK)
                        return result
                    except Exception as e:
                        span.set_status(StatusCode.ERROR, str(e))
                        span.record_exception(e)
                        raise
            return wrapper
        return decorator

    def tool(self, name: str = None):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(self_inst, *args, **kwargs):
                if not self._is_enabled():
                    return func(self_inst, *args, **kwargs)

                span_name = name or f"Tool:{func.__name__}"
                with self.tracer.start_as_current_span(span_name) as span:
                    span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.TOOL.value)

                    input_data = {"args": args, "kwargs": kwargs}
                    span.set_attribute(SpanAttributes.INPUT_VALUE, json.dumps(input_data, default=str))

                    try:
                        result = func(self_inst, *args, **kwargs)
                        span.set_attribute(SpanAttributes.OUTPUT_VALUE, str(result))
                        span.set_status(StatusCode.OK)
                        return result
                    except Exception as e:
                        span.set_status(StatusCode.ERROR, str(e))
                        span.record_exception(e)
                        raise
            return wrapper
        return decorator

    def stream(self, name: str = None):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(self_agent, *args, **kwargs):
                if not self._is_enabled():
                    yield from func(self_agent, *args, **kwargs)
                    return

                span_name = name or f"AgentStream:{self_agent.name}"
                with self.tracer.start_as_current_span(span_name) as span:
                    span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.AGENT.value)
                    self._inject_topology(span, self_agent)

                    full_content = []
                    try:
                        for chunk in func(self_agent, *args, **kwargs):
                            if hasattr(chunk, "content") and chunk.content:
                                full_content.append(str(chunk.content))
                            yield chunk

                        span.set_attribute(SpanAttributes.OUTPUT_VALUE, "".join(full_content))
                        span.set_status(StatusCode.OK)
                    except Exception as e:
                        span.set_status(StatusCode.ERROR, str(e))
                        span.record_exception(e)
                        raise
            return wrapper
        return decorator

    def astream(self, name: str = None):
        def decorator(func):
            @functools.wraps(func)
            async def wrapper(self_agent, *args, **kwargs):
                if not self._is_enabled():
                    async for chunk in func(self_agent, *args, **kwargs):
                        yield chunk
                    return

                span_name = name or f"AgentStream:{self_agent.name}"
                with self.tracer.start_as_current_span(span_name) as span:
                    span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.AGENT.value)
                    self._inject_topology(span, self_agent)

                    full_content = []
                    try:
                        async for chunk in func(self_agent, *args, **kwargs):
                            if hasattr(chunk, "content") and chunk.content:
                                full_content.append(str(chunk.content))
                            yield chunk

                        span.set_attribute(SpanAttributes.OUTPUT_VALUE, "".join(full_content))
                        span.set_status(StatusCode.OK)
                    except Exception as e:
                        span.set_status(StatusCode.ERROR, str(e))
                        span.record_exception(e)
                        raise
            return wrapper
        return decorator

    def flow(self, name: str = None):
        """Decorator to trace a full flow execution. Use it on function which orchestrates multiple agents/tools."""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if not self._is_enabled():
                    return func(*args, **kwargs)

                span_name = name or f"Flow:{func.__name__}"
                with self.tracer.start_as_current_span(span_name) as span:
                    span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "CHAIN")

                    try:
                        result = func(*args, **kwargs)
                        span.set_status(StatusCode.OK)
                        return result
                    except Exception as e:
                        span.set_status(StatusCode.ERROR, str(e))
                        span.record_exception(e)
                        raise
            return wrapper
        return decorator

Tracer = DecoratorTracer()
