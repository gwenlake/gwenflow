import atexit
import os
from typing import Dict

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from pydantic import BaseModel, Field, PrivateAttr


class TelemetryBase(BaseModel):
    service_name: str = Field(default="gwenflow-service")
    endpoint: str = Field(
        default_factory=lambda: os.getenv("OPENTELEMETRY_ENDPOINT", "http://localhost:4318/v1/traces")
    )
    headers: Dict[str, str] = Field(default_factory=dict)
    enabled: bool = Field(default=True)

    _tracer = PrivateAttr(default=None)

    def initialize(self) -> None:
        if not self.enabled:
            return

        current_provider = trace.get_tracer_provider()

        if not hasattr(current_provider, "resource"):
            resource = Resource.create({"service.name": self.service_name})
            provider = TracerProvider(resource=resource)

            exporter = OTLPSpanExporter(endpoint=self.endpoint, headers=self.headers)

            processor = BatchSpanProcessor(exporter)
            provider.add_span_processor(processor)
            trace.set_tracer_provider(provider)
            atexit.register(provider.shutdown)

        self._tracer = trace.get_tracer(self.service_name)

    def get_tracer(self):
        if self._tracer is None:
            self.initialize()
        return self._tracer if self._tracer else trace.get_tracer("noop")
