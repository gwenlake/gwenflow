import base64
import os
import threading
from dataclasses import dataclass, field
from typing import Callable

from gwenflow.logger import logger
from gwenflow.telemetry import _semconv as sc
from gwenflow.telemetry._settings import (
    is_otel_available,
    set_propagate_baggage,
    set_report_llm_usage,
    set_tracing_enabled,
    should_instrument_http,
)
from gwenflow.version import __version__

_HTTP_TRACES_PATH = "/v1/traces"
_GRPC_DEFAULT_ENDPOINT = "http://localhost:4317"

_context_processor_providers: set[int] = set()
_context_processor_lock = threading.Lock()

_http_instrumented = False


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

    from gwenflow.telemetry.tracer import _telemetry_context, attribute_value

    class ContextAttributeProcessor(SpanProcessor):
        def on_start(self, span, parent_context=None) -> None:
            attrs = _telemetry_context.get()
            if not attrs:
                return
            for key, value in attrs.items():
                span.set_attribute(key, attribute_value(value))

        def force_flush(self, timeout_millis: int = 30_000) -> bool:
            return True

    provider.add_span_processor(ContextAttributeProcessor())
    _context_processor_providers.add(id(provider))


def instrument_http_clients() -> bool:
    global _http_instrumented

    if _http_instrumented:
        return True

    try:
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    except ImportError:
        logger.debug(
            "opentelemetry-instrumentation-httpx is not installed; outgoing requests carry no "
            "trace context, so a service called from here cannot attach its spans to this trace. "
            'Install it with: pip install "gwenflow[telemetry]"'
        )
        return False

    instrumentor = HTTPXClientInstrumentor()
    if instrumentor.is_instrumented_by_opentelemetry:
        logger.debug("httpx is already instrumented; gwenflow reuses that instrumentation and leaves it as it is.")
        return True

    try:
        instrumentor.instrument()
    except Exception as e:
        logger.debug(f"Could not instrument httpx, outgoing requests carry no trace context: {e}")
        return False

    _http_instrumented = True
    return True


def uninstrument_http_clients() -> None:
    """Undo `instrument_http_clients` — only what gwenflow itself instrumented."""
    global _http_instrumented

    if not _http_instrumented:
        return
    try:
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

        HTTPXClientInstrumentor().uninstrument()
    except Exception as e:
        logger.debug(f"Could not uninstrument httpx: {e}")
    finally:
        _http_instrumented = False


def build_authorization(api_key: str | tuple[str, str]) -> str:
    if isinstance(api_key, str):
        return f"Bearer {api_key}"
    key_id, secret = api_key
    return "Basic " + base64.b64encode(f"{key_id}:{secret}".encode()).decode()


def resolve_protocol(protocol: str | None) -> str:
    if protocol:
        return protocol.upper()

    for var in ("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", "OTEL_EXPORTER_OTLP_PROTOCOL"):
        raw = (os.getenv(var) or "").strip().lower()
        if not raw:
            continue
        if raw == "grpc":
            return "GRPC"
        if raw == "http/protobuf":
            return "HTTP"
        logger.warning(f"{var}={raw} is not a protocol gwenflow exports in; falling back to http/protobuf.")
        return "HTTP"

    return "HTTP"


def resolve_endpoint(protocol: str, endpoint: str | None) -> str:
    proto = protocol.upper()

    if not endpoint:
        signal = os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
        if signal:
            return signal

    if proto == "GRPC":
        return endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT") or _GRPC_DEFAULT_ENDPOINT

    if not endpoint:
        endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT") or "http://localhost:4318"

    endpoint = endpoint.rstrip("/")
    if not endpoint.endswith(_HTTP_TRACES_PATH):
        endpoint += _HTTP_TRACES_PATH
    return endpoint


@dataclass
class Telemetry:
    organization: str | None = None
    protocol: str | None = None
    endpoint: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    api_key: str | tuple[str, str] | None = None
    auth: Callable[[], dict[str, str]] | None = None
    service_name: str | None = None
    service_version: str | None = None
    insecure: bool | None = None
    report_llm_usage: bool | None = None
    instrument_http: bool | None = None
    propagate_baggage: bool | None = None

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
        if self.report_llm_usage is not None:
            set_report_llm_usage(self.report_llm_usage)
        if self.propagate_baggage is not None:
            set_propagate_baggage(self.propagate_baggage)
        self._has_export_config = bool(self.endpoint or self.api_key or self.auth or self.headers)
        self.protocol = resolve_protocol(self.protocol)
        self.endpoint = resolve_endpoint(self.protocol, self.endpoint)
        self._configure()

    def _build_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = build_authorization(self.api_key)
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
            self._enable()
            return

        resource = Resource.create(
            build_resource_attributes(self.organization, self.service_name, self.service_version)
        )
        provider = TracerProvider(resource=resource)
        provider.add_span_processor(BatchSpanProcessor(self._build_exporter()))
        _install_context_processor(provider)
        trace.set_tracer_provider(provider)
        self._enable()
        logger.debug(
            f"Telemetry enabled (organization={self.organization}, protocol={self.protocol}, endpoint={self.endpoint})."
        )

    def _enable(self) -> None:
        set_tracing_enabled(True)
        wanted = self.instrument_http if self.instrument_http is not None else should_instrument_http()
        if wanted:
            instrument_http_clients()

    def _build_exporter(self):
        headers = self._build_headers() or None
        if self.protocol.upper() == "GRPC":
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

            return OTLPSpanExporter(endpoint=self.endpoint, headers=headers, insecure=self.insecure)

        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        return OTLPSpanExporter(endpoint=self.endpoint, headers=headers)

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
