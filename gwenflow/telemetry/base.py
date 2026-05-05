import atexit
import os
from dataclasses import dataclass, field


@dataclass
class Telemetry:
    service_name: str = field(default_factory=lambda: os.getenv("OTEL_SERVICE_NAME", "gwenflow-service"))
    protocol: str = "HTTP"
    endpoint: str | None = None
    headers: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.endpoint:
            if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
                self.endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
            elif self.protocol.upper() == "GRPC":
                self.endpoint = "http://localhost:4317"
            else:
                self.endpoint = "http://localhost:4318/v1/traces"

        try:
            from opentelemetry import trace
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
        except ImportError as exc:
            raise ImportError(
                "OpenTelemetry is not installed. "
                "Run: uv add opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp openinference-semantic-conventions"
            ) from exc

        if isinstance(trace.get_tracer_provider(), TracerProvider):
            return

        resource = Resource.create({"service.name": self.service_name})
        provider = TracerProvider(resource=resource)

        if self.protocol.upper() == "GRPC":
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        else:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=self.endpoint, headers=self.headers)))
        trace.set_tracer_provider(provider)
        atexit.register(provider.shutdown)
