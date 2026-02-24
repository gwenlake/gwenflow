import atexit
import os
from typing import Any, Dict, Optional

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from pydantic import BaseModel, Field, PrivateAttr, model_validator


class TelemetryBase(BaseModel):
    service_name: str = "gwenflow-service"
    protocol: str = "HTTP"
    endpoint: Optional[str] = None
    headers: Dict[str, str] = Field(default_factory=dict)
    enabled: bool = True

    _tracer: Any = PrivateAttr(default=None)

    @model_validator(mode='after')
    def set_default_endpoint(self) -> 'TelemetryBase':
        if not self.endpoint:
            env_endpoint = os.getenv("TELEMETRY_ENDPOINT")
            if env_endpoint:
                self.endpoint = env_endpoint
            else:
                if self.protocol.upper() == "HTTP":
                    self.endpoint = "http://localhost:6006/v1/traces"
                else:
                    self.endpoint = "http://localhost:4317" # For gRPC
        return self

    def initialize(self) -> None:
        if not self.enabled:
            return

        current_provider = trace.get_tracer_provider()

        if not isinstance(current_provider, TracerProvider):

            resource = Resource.create({"service.name": self.service_name})
            provider = TracerProvider(resource=resource)

            if self.protocol.upper() == "GRPC":
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter as GRPCOTLPExporter
                exporter = GRPCOTLPExporter(endpoint=self.endpoint, headers=self.headers)
            else:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as HTTPOTLPExporter
                exporter = HTTPOTLPExporter(endpoint=self.endpoint, headers=self.headers)

            processor = BatchSpanProcessor(exporter)
            provider.add_span_processor(processor)
            trace.set_tracer_provider(provider)
            atexit.register(provider.shutdown)

        self._tracer = trace.get_tracer(self.service_name)

    def get_tracer(self):
        if self._tracer is None:
            self.initialize()
        return self._tracer if self._tracer else trace.get_tracer("noop")

