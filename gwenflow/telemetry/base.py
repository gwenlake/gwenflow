import atexit
import os
from typing import Dict, Optional

from pydantic import BaseModel, Field, model_validator


class TelemetryBase(BaseModel):
    service_name: str = "gwenflow-service"
    protocol: str = "HTTP"
    endpoint: Optional[str] = None
    headers: Dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def set_default_endpoint(self) -> "TelemetryBase":
        if not self.endpoint:
            env_endpoint = os.getenv("TELEMETRY_ENDPOINT")
            if env_endpoint:
                self.endpoint = env_endpoint
            else:
                if self.protocol.upper() == "HTTP":
                    self.endpoint = "http://localhost:4318"
                else:
                    self.endpoint = "localhost:4317"  # For gRPC
        return self

    def initialize(self) -> None:
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
        except ImportError as exc:
            raise ImportError(
                "Opentelemetry is not installed.\n"
                "To enable it, install the required packages: "
                "`uv add opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp openinference-semantic-conventions`"
            ) from exc

        current_provider = trace.get_tracer_provider()

        if not isinstance(current_provider, TracerProvider):
            resource = Resource.create(
                {
                    "service.name": self.service_name,
                }
            )
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
