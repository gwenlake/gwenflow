import atexit
import os
from dataclasses import dataclass, field
from typing import Callable

from gwenflow.logger import logger
from gwenflow.telemetry._settings import is_otel_available, set_tracing_enabled
from gwenflow.version import __version__

_HTTP_TRACES_PATH = "/v1/traces"


def build_resource_attributes(organization: str | None) -> dict[str, str]:
    attributes = {"service.version": __version__}
    if organization:
        attributes["service.name"] = organization
    return attributes


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
    api_key: str | None = None
    auth: Callable[[], dict[str, str]] | None = None

    def __post_init__(self) -> None:
        if self.api_key is None:
            self.api_key = os.getenv("GWENFLOW_TELEMETRY_API_KEY")
        if self.organization is None:
            self.organization = os.getenv("OTEL_SERVICE_NAME")
            if self.organization is None and self.api_key is None:
                self.organization = "gwenflow"
        self._has_export_config = bool(self.endpoint or self.api_key or self.auth or self.headers)
        self.endpoint = resolve_endpoint(self.protocol, self.endpoint)
        self._configure()

    def _build_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.auth is not None:
            headers.update(self.auth() or {})
        headers.update(self.headers)
        return headers

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
            if self._has_export_config:
                logger.warning(
                    "A TracerProvider is already configured; gwenflow reuses it and the "
                    "endpoint/api_key/auth/headers passed to Telemetry() are ignored. "
                    "Configure the export on the existing provider, or call Telemetry() only once."
                )
            else:
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
        return OTLPSpanExporter(endpoint=self.endpoint, headers=self._build_headers() or None)
