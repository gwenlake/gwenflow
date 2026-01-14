from typing import Dict, Optional

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from pydantic import BaseModel, ConfigDict, Field


class TelemetryBase(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    service_name: str = Field(default="gwenflow-service")
    tags: Dict[str, str] = Field(default_factory=dict)
    current_provider: Optional[TracerProvider] = None
    endpoint: str = Field(default="http://localhost:6006/v1/traces")
    enabled: bool = Field(default=True)

    def setup_telemetry(self) -> TracerProvider:
        if not self.enabled:
            return None

        resource_attributes = {"service.name": self.service_name}
        resource_attributes.update(self.tags)

        provider = trace.get_tracer_provider()

        if not hasattr(provider, "resource") or provider.resource.attributes.get("service.name") == "unknown_service":
            resource = Resource.create(resource_attributes)
            provider = TracerProvider(resource=resource)
            trace.set_tracer_provider(provider)

        self.current_provider = provider
        return provider

    def add_exporter(self):
        if self.enabled and self.current_provider:
            exporter = OTLPSpanExporter(endpoint=self.endpoint)
            processor = BatchSpanProcessor(exporter)
            self.current_provider.add_span_processor(processor)
