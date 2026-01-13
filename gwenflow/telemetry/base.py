import functools
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from openinference.semconv.trace import SpanAttributes, OpenInferenceSpanKindValues


class TelemetryBase(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    service_name: str = Field(default="gwenflow-service")
    endpoint: str = Field(default="http://localhost:6006/v1/traces")
    current_provider: Optional[TracerProvider] = None

    def setup_telemetry(self) -> TracerProvider:
        provider = trace.get_tracer_provider()
        if not hasattr(provider, "resource"):
            resource = Resource.create({"service.name": self.service_name})
            provider = TracerProvider(resource=resource)
            trace.set_tracer_provider(provider)
        
        self.current_provider = provider
        return provider
    
    def add_phoenix_exporter(self): #TODO create own class for Phoenix exporter
        if self.current_provider:
            exporter = OTLPSpanExporter(endpoint=self.endpoint)
            processor = BatchSpanProcessor(exporter)
            self.current_provider.add_span_processor(processor)


tracer = trace.get_tracer("gwenflow")

def trace_agent(name: str = None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            span_name = name or f"Agent:{self.name}"
            with tracer.start_as_current_span(span_name) as span:
                span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.AGENT.value)

                span.set_attribute("agent.id", str(self.id))
                span.set_attribute("llm.model_name", getattr(self.llm, 'model', 'unknown'))
                
                query = kwargs.get("query") or (args[0] if args else "None")
                span.set_attribute(SpanAttributes.INPUT_VALUE, str(query))
                try:
                    result = func(self, *args, **kwargs)
                    span.set_attribute(SpanAttributes.OUTPUT_VALUE, str(result))
                    return result
                except Exception as e:
                    span.record_exception(e)
                    raise
        return wrapper
    return decorator

def trace_tool(name: str = None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            span_name = name or func.__name__
            with tracer.start_as_current_span(span_name) as span:
                span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.TOOL.value)
                span.set_attribute(SpanAttributes.INPUT_VALUE, str(args) + str(kwargs))
                result = func(*args, **kwargs)
                span.set_attribute(SpanAttributes.OUTPUT_VALUE, str(result))
                return result
        return wrapper
    return decorator