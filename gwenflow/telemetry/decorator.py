import functools

from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
from opentelemetry import trace


class DecoratorTracer:
    def __init__(self, tracer_name: str = "gwenflow"):
        self.tracer = trace.get_tracer(tracer_name)

    def _is_enabled(self) -> bool:
        """Vérifie si un provider avec une ressource valide est configuré."""
        provider = trace.get_tracer_provider()
        return hasattr(provider, "resource") and provider.resource.attributes.get("service.name") != "unknown_service"

    def agent(self, name: str = None):
        """Décorateur pour les méthodes run() classiques des agents."""
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

                    query = kwargs.get("query") or (args[0] if args else "None")
                    span.set_attribute(SpanAttributes.INPUT_VALUE, str(query))

                    try:
                        result = func(self_agent, *args, **kwargs)
                        span.set_attribute(SpanAttributes.OUTPUT_VALUE, str(result))
                        return result
                    except Exception as e:
                        span.record_exception(e)
                        raise
            return wrapper
        return decorator

    def tool(self, name: str = None):
        """Décorateur pour l'exécution des outils."""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(self_inst, *args, **kwargs):
                if not self._is_enabled():
                    return func(self_inst, *args, **kwargs)

                span_name = name or f"Tool:{func.__name__}"
                with self.tracer.start_as_current_span(span_name) as span:
                    span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.TOOL.value)
                    span.set_attribute(SpanAttributes.INPUT_VALUE, str(args) + str(kwargs))

                    result = func(self_inst, *args, **kwargs)
                    span.set_attribute(SpanAttributes.OUTPUT_VALUE, str(result))
                    return result
            return wrapper
        return decorator

    def stream(self, name: str = None):
        """Décorateur pour le streaming synchrone (Iterator)."""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(self_agent, *args, **kwargs):
                if not self._is_enabled():
                    yield from func(self_agent, *args, **kwargs)
                    return

                span_name = name or f"AgentStream:{self_agent.name}"
                with self.tracer.start_as_current_span(span_name) as span:
                    span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.AGENT.value)
                    full_content = []
                    for chunk in func(self_agent, *args, **kwargs):
                        if hasattr(chunk, "content") and chunk.content:
                            full_content.append(str(chunk.content))
                        yield chunk
                    span.set_attribute(SpanAttributes.OUTPUT_VALUE, "".join(full_content))
            return wrapper
        return decorator

    def astream(self, name: str = None):
        """Décorateur pour le streaming asynchrone (AsyncIterator)."""
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
                    full_content = []
                    async for chunk in func(self_agent, *args, **kwargs):
                        if hasattr(chunk, "content") and chunk.content:
                            full_content.append(str(chunk.content))
                        yield chunk
                    span.set_attribute(SpanAttributes.OUTPUT_VALUE, "".join(full_content))
            return wrapper
        return decorator

Tracer = DecoratorTracer()
