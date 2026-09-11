import importlib.util
import os

_tracing_enabled: bool = False


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False


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


def is_otel_available() -> bool:
    return _has_module("opentelemetry.sdk.trace") and _has_module("opentelemetry.exporter.otlp")


def is_tracing_enabled() -> bool:
    return _tracing_enabled


def set_tracing_enabled(value: bool) -> None:
    global _tracing_enabled
    _tracing_enabled = value


_report_llm_usage: bool | None = None


def should_report_llm_usage() -> bool:
    """Whether this process is the one that measures its own LLM calls.

    Set it False when the calls are served by a remote endpoint that emits a
    span for them as well — a gateway, a proxy, an inference service. The call
    is then described by two spans, gwenflow's and the server's, and both would
    carry `llm.token_count.*` for the same tokens. Any backend that aggregates
    that attribute across spans reads it twice: twice the tokens, twice the
    cost, twice whatever it derives from them.

    Which of the two should report is not a question this library can answer —
    it depends on which end you trust to know what actually served the call, so
    it is left to whoever wires the two together. Everything else about the
    span is unaffected: it still says a model was called, with what, for how
    long, and how it ended.
    """
    if _report_llm_usage is not None:
        return _report_llm_usage
    return _env_bool("GWENFLOW_TELEMETRY_REPORT_LLM_USAGE", True)


def set_report_llm_usage(value: bool | None) -> None:
    """`None` hands the decision back to the environment."""
    global _report_llm_usage
    _report_llm_usage = value


def should_instrument_http() -> bool:
    return _env_bool("GWENFLOW_TELEMETRY_INSTRUMENT_HTTP", True)


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
    if limit <= 0 or len(value) <= limit:
        return value

    marker = f"... [truncated {len(value) - limit} chars] ..."
    head = limit // 2
    tail = limit - head
    return value[:head] + marker + value[-tail:]
