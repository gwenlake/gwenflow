import atexit
import os
from dataclasses import dataclass, field

from gwenflow.logger import logger
from gwenflow.telemetry._settings import is_otel_available, set_tracing_enabled
from gwenflow.version import __version__

_HTTP_TRACES_PATH = "/v1/traces"


def build_resource_attributes(organization: str) -> dict[str, str]:
    return {
        "service.name": organization,
        "service.version": __version__,
    }


def resolve_endpoint(protocol: str, endpoint: str | None) -> str:
    proto = protocol.upper()

    if not endpoint:
        signal = os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
        if signal:
            return signal

    if proto == "GRPC":
        return endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT") or "localhost:4317"

    if not endpoint:
        endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT") or "http://localhost:4318"

    endpoint = endpoint.rstrip("/")
    if not endpoint.endswith(_HTTP_TRACES_PATH):
        endpoint += _HTTP_TRACES_PATH
    return endpoint


@dataclass
class Telemetry:
    organization: str | None = None
    protocol: str = "HTTP"
    endpoint: str | None = None
    headers: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.organization is None:
            self.organization = os.getenv("OTEL_SERVICE_NAME", "gwenflow")
        self.endpoint = resolve_endpoint(self.protocol, self.endpoint)
        self._configure()

    def _configure(self) -> None:
        if os.getenv("OTEL_SDK_DISABLED", "").strip().lower() == "true":
            logger.info("Telemetry disabled via OTEL_SDK_DISABLED; skipping setup.")
            return

        if not is_otel_available():
            logger.warning(
                "OpenTelemetry packages are not installed; telemetry is disabled. "
                'Enable it with: pip install "gwenflow[telemetry]"'
            )
            return

        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        if isinstance(trace.get_tracer_provider(), TracerProvider):
            logger.debug("A TracerProvider is already configured; reusing it for gwenflow telemetry.")
            set_tracing_enabled(True)
            return

        resource = Resource.create(build_resource_attributes(self.organization))
        provider = TracerProvider(resource=resource)
        provider.add_span_processor(BatchSpanProcessor(self._build_exporter()))
        trace.set_tracer_provider(provider)
        atexit.register(provider.shutdown)
        set_tracing_enabled(True)
        logger.debug(
            f"Telemetry enabled (organization={self.organization}, protocol={self.protocol}, endpoint={self.endpoint})."
        )

    def _build_exporter(self):
        if self.protocol.upper() == "GRPC":
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        else:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        return OTLPSpanExporter(endpoint=self.endpoint, headers=self.headers or None)
