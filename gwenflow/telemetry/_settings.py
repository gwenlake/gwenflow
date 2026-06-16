import importlib.util
import os

_tracing_enabled: bool = False


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False


def is_otel_available() -> bool:
    return _has_module("opentelemetry.sdk.trace") and _has_module("opentelemetry.exporter.otlp")


def is_openinference_available() -> bool:
    return _has_module("openinference.semconv.trace")


def is_tracing_enabled() -> bool:
    return _tracing_enabled


def set_tracing_enabled(value: bool) -> None:
    global _tracing_enabled
    _tracing_enabled = value


_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    return default


def _capture_content() -> bool:
    return _env_bool("GWENFLOW_TELEMETRY_CAPTURE_CONTENT", True)


def should_capture_inputs() -> bool:
    return _capture_content() and not _env_bool("OPENINFERENCE_HIDE_INPUTS", False)


def should_capture_outputs() -> bool:
    return _capture_content() and not _env_bool("OPENINFERENCE_HIDE_OUTPUTS", False)


def max_attribute_length() -> int:
    raw = os.getenv("GWENFLOW_TELEMETRY_MAX_ATTR_LENGTH")
    if raw is None:
        return 8192
    try:
        return int(raw)
    except ValueError:
        return 8192


REDACTED = "__REDACTED__"


def truncate(value: str) -> str:
    limit = max_attribute_length()
    if limit > 0 and len(value) > limit:
        return value[:limit] + f"... [truncated {len(value) - limit} chars]"
    return value
