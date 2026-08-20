import os
import threading
from dataclasses import dataclass, field
from typing import Callable

from gwenflow.logger import logger
from gwenflow.telemetry import _semconv as sc
from gwenflow.telemetry._settings import is_otel_available, set_tracing_enabled
from gwenflow.version import __version__

_HTTP_TRACES_PATH = "/v1/traces"

_context_processor_providers: set[int] = set()
_context_processor_lock = threading.Lock()


def build_resource_attributes(
    organization: str | None,
    service_name: str | None = None,
    service_version: str | None = None,
) -> dict[str, str]:
    attributes = {sc.DISTRO_NAME: "gwenflow", sc.DISTRO_VERSION: __version__}
    name = service_name or organization
    if name:
        attributes[sc.SERVICE_NAME] = name
    if organization:
        attributes[sc.ORGANIZATION] = organization
    if service_version:
        attributes[sc.SERVICE_VERSION] = service_version
    return attributes


def _install_context_processor(provider) -> None:
    with _context_processor_lock:
        _install_context_processor_locked(provider)


def _install_context_processor_locked(provider) -> None:
    if id(provider) in _context_processor_providers:
        return

    from opentelemetry.sdk.trace import SpanProcessor

    from gwenflow.telemetry.tracer import _telemetry_context

    class ContextAttributeProcessor(SpanProcessor):
        def on_start(self, span, parent_context=None) -> None:
            attrs = _telemetry_context.get()
            if not attrs:
                return
            for key, value in attrs.items():
                span.set_attribute(key, value if isinstance(value, (str, bool, int, float)) else str(value))

        def force_flush(self, timeout_millis: int = 30_000) -> bool:
            # No export to flush here; the base SpanProcessor implementation
            # returns None, which would make TracerProvider.force_flush()
            # (a logical AND across all processors) report failure even when
            # the real exporter (e.g. BatchSpanProcessor) succeeded.
            return True

    provider.add_span_processor(ContextAttributeProcessor())
    _context_processor_providers.add(id(provider))


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
    service_name: str | None = None
    service_version: str | None = None

    def __post_init__(self) -> None:
        if self.api_key is None:
            self.api_key = os.getenv("GWENFLOW_TELEMETRY_API_KEY")
        if self.service_name is None:
            self.service_name = os.getenv("OTEL_SERVICE_NAME")
        if self.service_version is None:
            self.service_version = os.getenv("OTEL_SERVICE_VERSION")
        if self.organization is None:
            self.organization = os.getenv("GWENFLOW_ORGANIZATION")
        if self.service_name is None and self.organization is None and self.api_key is None:
            self.service_name = "gwenflow"
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

        existing = trace.get_tracer_provider()
        if isinstance(existing, TracerProvider):
            if self._has_export_config:
                logger.warning(
                    "A TracerProvider is already configured; gwenflow reuses it and the "
                    "endpoint/api_key/auth/headers passed to Telemetry() are ignored. "
                    "Configure the export on the existing provider, or call Telemetry() only once."
                )
            else:
                logger.debug("A TracerProvider is already configured; reusing it for gwenflow telemetry.")
            _install_context_processor(existing)
            set_tracing_enabled(True)
            return

        resource = Resource.create(
            build_resource_attributes(self.organization, self.service_name, self.service_version)
        )
        provider = TracerProvider(resource=resource)
        provider.add_span_processor(BatchSpanProcessor(self._build_exporter()))
        _install_context_processor(provider)
        trace.set_tracer_provider(provider)
        # No atexit.register here: TracerProvider(shutdown_on_exit=True), the
        # default, already registers its own shutdown handler.
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

    @staticmethod
    def _sdk_provider():
        if not is_otel_available():
            return None
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider

        provider = trace.get_tracer_provider()
        return provider if isinstance(provider, TracerProvider) else None

    def flush(self, timeout_millis: int = 30_000) -> bool:
        """Export everything still queued. Call it on shutdown: `atexit` does not run on SIGKILL."""
        provider = self._sdk_provider()
        return bool(provider.force_flush(timeout_millis)) if provider else False

    def shutdown(self) -> None:
        provider = self._sdk_provider()
        if provider:
            provider.shutdown()
