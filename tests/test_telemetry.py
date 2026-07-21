import asyncio
import contextvars

import pytest

import gwenflow.telemetry.base as base_mod
from gwenflow.telemetry._settings import is_tracing_enabled, set_tracing_enabled
from gwenflow.telemetry.base import Telemetry, build_resource_attributes, resolve_endpoint
from gwenflow.telemetry.tracer import DecoratorTracer
from gwenflow.types import Document

pytest.importorskip("opentelemetry.sdk.trace")

from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter  # noqa: E402
from opentelemetry.trace import StatusCode  # noqa: E402

T = DecoratorTracer("gwenflow-test")


class _Usage:
    input_tokens = 12
    output_tokens = 8
    total_tokens = 20
    cache_read_tokens = 3
    cache_write_tokens = 0
    input_audio_tokens = 4
    output_audio_tokens = 2
    details = {"reasoning_tokens": 5}


class _Response:
    content = "hello world"
    tool_calls: list = []
    usage = _Usage()
    finish_reason = "stop"


class _Chunk:
    def __init__(self, content):
        self.content = content
        self.tool_calls = None


class FakeLLM:
    model = "gpt-test"
    _model_params = {"temperature": 0.0}

    @T.llm(name="LLM Invoke")
    def invoke(self, input):
        return _Response()

    @T.llm(name="LLM Async Invoke")
    async def ainvoke(self, input):
        return _Response()

    @T.llm(name="LLM Stream")
    def stream(self, input):
        for piece in ("hello ", "world"):
            yield _Chunk(piece)

    @T.llm(name="LLM Astream")
    async def astream(self, input):
        for piece in ("hello ", "world"):
            yield _Chunk(piece)

    @T.llm(name="LLM Boom")
    def boom(self, input):
        raise ValueError("kaboom")


class FakeAgent:
    name = "Researcher"

    def __init__(self):
        self.llm = FakeLLM()

    @T.agent(name="Agent Run")
    def run(self, input):
        return self.llm.invoke(input)


class _AgentUsage:
    requests = 2
    tool_calls = 1


class _AgentResult:
    content = "done"
    tool_calls = None
    usage = _AgentUsage()


class FakeAgentWithUsage:
    name = "Planner"

    @T.agent(name="Agent Run With Usage")
    def run(self, input):
        return _AgentResult()


class FakeRetriever:
    name = "kb"

    @T.retriever(name="Retriever Search")
    def search(self, query):
        return [
            Document(id="d1", content="alpha", score=0.9, metadata={"src": "wiki"}),
            Document(id="d2", content="beta", score=0.5),
        ]


class FakeEmbeddings:
    model = "emb-test"

    @T.embedding(name="Embed Documents")
    def embed_documents(self, texts):
        return [[0.1, 0.2] for _ in texts]


class FakeReranker:
    model = "rr-test"
    top_k = 3

    @T.reranker(name="Rerank")
    def rerank(self, query, documents):
        return documents


@pytest.fixture
def spans():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    T._tracer = provider.get_tracer("test")
    set_tracing_enabled(True)
    try:
        yield exporter
    finally:
        set_tracing_enabled(False)
        T._tracer = None
        exporter.clear()


# --- resolve_endpoint -----------------------------------------------------------


@pytest.mark.parametrize(
    "endpoint,expected",
    [
        ("http://localhost:4318", "http://localhost:4318/v1/traces"),
        ("http://localhost:4318/", "http://localhost:4318/v1/traces"),
        ("http://localhost:4318/v1/traces", "http://localhost:4318/v1/traces"),
        ("https://collector.example.com:443", "https://collector.example.com:443/v1/traces"),
    ],
)
def test_resolve_endpoint_http(endpoint, expected):
    assert resolve_endpoint("HTTP", endpoint) == expected


def test_resolve_endpoint_grpc_is_verbatim():
    assert resolve_endpoint("GRPC", "localhost:4317") == "localhost:4317"


def test_resolve_endpoint_http_default():
    assert resolve_endpoint("HTTP", None) == "http://localhost:4318/v1/traces"


def test_resolve_endpoint_grpc_default():
    assert resolve_endpoint("GRPC", None) == "localhost:4317"


def test_resolve_endpoint_reads_base_env(monkeypatch):
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4318")
    assert resolve_endpoint("HTTP", None) == "http://collector:4318/v1/traces"


def test_resolve_endpoint_per_signal_env_is_verbatim(monkeypatch):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://collector:4318/custom")
    assert resolve_endpoint("HTTP", None) == "http://collector:4318/custom"


# --- resource / organization ----------------------------------------------------


def test_organization_maps_to_service_name():
    attrs = build_resource_attributes("acme-corp")
    assert attrs["service.name"] == "acme-corp"
    assert attrs["service.version"]
    assert "openinference.project.name" not in attrs


def test_resource_omits_service_name_without_organization():
    attrs = build_resource_attributes(None)
    assert "service.name" not in attrs
    assert attrs["service.version"]


def test_api_key_overrides_default_organization(monkeypatch):
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    monkeypatch.delenv("OTEL_SERVICE_NAME", raising=False)
    t = Telemetry(api_key="secret-123")
    assert t.organization is None


def test_default_organization_without_api_key(monkeypatch):
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    monkeypatch.delenv("OTEL_SERVICE_NAME", raising=False)
    monkeypatch.delenv("GWENFLOW_TELEMETRY_API_KEY", raising=False)
    t = Telemetry()
    assert t.organization == "gwenflow"


def test_explicit_organization_kept_even_with_api_key(monkeypatch):
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    t = Telemetry(organization="acme", api_key="secret-123")
    assert t.organization == "acme"


# --- enablement / no-op ---------------------------------------------------------


def test_telemetry_noop_when_deps_missing(monkeypatch):
    set_tracing_enabled(False)
    monkeypatch.setattr(base_mod, "is_otel_available", lambda: False)
    Telemetry(organization="x")
    assert is_tracing_enabled() is False


def test_telemetry_disabled_via_env(monkeypatch):
    set_tracing_enabled(False)
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    Telemetry(organization="x")
    assert is_tracing_enabled() is False


def _reuse_existing_provider(monkeypatch):
    """Force _configure() down the 'a TracerProvider already exists' branch."""
    from opentelemetry import trace as otel_trace

    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
    monkeypatch.delenv("GWENFLOW_TELEMETRY_API_KEY", raising=False)
    monkeypatch.setattr(base_mod, "is_otel_available", lambda: True)
    existing = TracerProvider()  # a genuine SDK provider, never set globally
    monkeypatch.setattr(otel_trace, "get_tracer_provider", lambda: existing)


def test_reuse_existing_provider_warns_when_export_config_ignored(monkeypatch):
    _reuse_existing_provider(monkeypatch)
    warnings: list[str] = []
    monkeypatch.setattr(base_mod.logger, "warning", lambda msg, *a, **k: warnings.append(msg))

    Telemetry(organization="acme", api_key="secret-123")

    assert any("ignored" in w for w in warnings)


def test_reuse_existing_provider_is_quiet_without_export_config(monkeypatch):
    _reuse_existing_provider(monkeypatch)
    warnings: list[str] = []
    monkeypatch.setattr(base_mod.logger, "warning", lambda msg, *a, **k: warnings.append(msg))

    Telemetry(organization="acme")  # no endpoint/api_key/auth/headers to drop

    assert warnings == []
    assert is_tracing_enabled() is True


# --- api key / auth headers -----------------------------------------------------


def test_api_key_sets_bearer_authorization_header(monkeypatch):
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    t = Telemetry(organization="acme", api_key="secret-123")
    assert t._build_headers() == {"Authorization": "Bearer secret-123"}


def test_no_api_key_means_no_auth_header(monkeypatch):
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    monkeypatch.delenv("GWENFLOW_TELEMETRY_API_KEY", raising=False)
    t = Telemetry(organization="acme")
    assert t._build_headers() == {}


def test_api_key_merges_with_custom_headers(monkeypatch):
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    t = Telemetry(organization="acme", api_key="secret", headers={"x-tenant": "acme"})
    headers = t._build_headers()
    assert headers["x-tenant"] == "acme"
    assert headers["Authorization"] == "Bearer secret"


def test_explicit_authorization_header_wins_over_api_key(monkeypatch):
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    t = Telemetry(organization="acme", api_key="secret", headers={"Authorization": "Bearer custom"})
    assert t._build_headers()["Authorization"] == "Bearer custom"


def test_api_key_read_from_env(monkeypatch):
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    monkeypatch.setenv("GWENFLOW_TELEMETRY_API_KEY", "env-key")
    t = Telemetry(organization="acme")
    assert t._build_headers() == {"Authorization": "Bearer env-key"}


def test_auth_callable_provides_arbitrary_scheme(monkeypatch):
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    monkeypatch.delenv("GWENFLOW_TELEMETRY_API_KEY", raising=False)
    t = Telemetry(organization="acme", auth=lambda: {"x-api-key": "k", "x-tenant": "acme"})
    assert t._build_headers() == {"x-api-key": "k", "x-tenant": "acme"}


def test_auth_overrides_api_key_sugar(monkeypatch):
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    t = Telemetry(organization="acme", api_key="sugar", auth=lambda: {"Authorization": "Custom xyz"})
    assert t._build_headers()["Authorization"] == "Custom xyz"


def test_explicit_headers_win_over_auth(monkeypatch):
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    t = Telemetry(
        organization="acme",
        auth=lambda: {"Authorization": "Bearer from-auth"},
        headers={"Authorization": "Bearer explicit"},
    )
    assert t._build_headers()["Authorization"] == "Bearer explicit"


# --- no-op gate -----------------------------------------------------------------


def test_disabled_is_passthrough_and_emits_nothing():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    T._tracer = provider.get_tracer("test")
    set_tracing_enabled(False)  # explicitly off
    try:
        assert FakeLLM().invoke("hi").content == "hello world"
        assert list(FakeLLM().stream("hi"))  # generator still works
        assert exporter.get_finished_spans() == ()
    finally:
        T._tracer = None


# --- span content ---------------------------------------------------------------


def test_llm_span_attributes(spans):
    FakeLLM().invoke("what is 2+2?")
    (span,) = spans.get_finished_spans()
    attrs = dict(span.attributes)
    assert span.name == "LLM Invoke"
    assert attrs["openinference.span.kind"] == "LLM"
    assert attrs["llm.model_name"] == "gpt-test"
    assert attrs["input.value"] == "what is 2+2?"
    assert attrs["output.value"] == "hello world"
    assert attrs["llm.token_count.prompt"] == 12
    assert attrs["llm.token_count.completion"] == 8
    assert attrs["llm.token_count.total"] == 20
    assert attrs["llm.token_count.prompt_details.cache_read"] == 3
    assert span.status.status_code == StatusCode.OK


def test_llm_span_captures_finish_reason_and_detailed_tokens(spans):
    FakeLLM().invoke("hi")
    attrs = dict(spans.get_finished_spans()[0].attributes)
    assert attrs["llm.finish_reason"] == "stop"
    assert attrs["llm.token_count.prompt_details.audio"] == 4
    assert attrs["llm.token_count.completion_details.audio"] == 2
    assert attrs["llm.token_count.completion_details.reasoning"] == 5


def test_agent_span_captures_run_counts(spans):
    FakeAgentWithUsage().run("go")
    attrs = dict(spans.get_finished_spans()[0].attributes)
    assert attrs["gwenflow.agent.llm_requests"] == 2
    assert attrs["gwenflow.agent.tool_calls"] == 1


# --- RAG span kinds (retriever / embedding / reranker) --------------------------


def test_retriever_span_captures_documents(spans):
    FakeRetriever().search("what is rag?")
    (span,) = spans.get_finished_spans()
    attrs = dict(span.attributes)
    assert span.name == "Retriever Search"
    assert attrs["openinference.span.kind"] == "RETRIEVER"
    assert attrs["input.value"] == "what is rag?"
    assert attrs["retrieval.documents.0.document.id"] == "d1"
    assert attrs["retrieval.documents.0.document.content"] == "alpha"
    assert attrs["retrieval.documents.0.document.score"] == 0.9
    assert attrs["retrieval.documents.1.document.id"] == "d2"
    assert span.status.status_code == StatusCode.OK


def test_retriever_redacts_document_content_but_keeps_ids(spans, monkeypatch):
    monkeypatch.setenv("GWENFLOW_TELEMETRY_CAPTURE_CONTENT", "false")
    FakeRetriever().search("q")
    attrs = dict(spans.get_finished_spans()[0].attributes)
    assert attrs["retrieval.documents.0.document.content"] == "__REDACTED__"
    assert attrs["retrieval.documents.0.document.id"] == "d1"
    assert attrs["retrieval.documents.0.document.score"] == 0.9


def test_embedding_span(spans):
    FakeEmbeddings().embed_documents(["a", "b"])
    (span,) = spans.get_finished_spans()
    attrs = dict(span.attributes)
    assert span.name == "Embed Documents"
    assert attrs["openinference.span.kind"] == "EMBEDDING"
    assert attrs["embedding.model_name"] == "emb-test"
    # Raw vectors are never emitted as an output.
    assert "output.value" not in attrs


def test_reranker_span(spans):
    docs = [Document(content="a"), Document(content="b")]
    FakeReranker().rerank("q", docs)
    (span,) = spans.get_finished_spans()
    attrs = dict(span.attributes)
    assert span.name == "Rerank"
    assert attrs["openinference.span.kind"] == "RERANKER"
    assert attrs["reranking.model_name"] == "rr-test"
    assert attrs["reranking.top_k"] == 3
    assert attrs["reranking.output_documents.0.document.content"] == "a"


def test_agent_wraps_llm_with_correct_nesting(spans):
    FakeAgent().run("find the score")
    finished = spans.get_finished_spans()
    by_name = {s.name: s for s in finished}
    assert set(by_name) == {"Agent Run", "LLM Invoke"}
    agent_span, llm_span = by_name["Agent Run"], by_name["LLM Invoke"]
    assert dict(agent_span.attributes)["gwenflow.agent.name"] == "Researcher"
    assert llm_span.parent is not None
    assert llm_span.parent.span_id == agent_span.context.span_id
    assert llm_span.context.trace_id == agent_span.context.trace_id


def test_streaming_accumulates_output(spans):
    list(FakeLLM().stream("go"))
    (span,) = spans.get_finished_spans()
    assert span.name == "LLM Stream"
    assert dict(span.attributes)["output.value"] == "hello world"
    assert span.status.status_code == StatusCode.OK


def test_async_invoke(spans):
    asyncio.run(FakeLLM().ainvoke("hi"))
    (span,) = spans.get_finished_spans()
    assert span.name == "LLM Async Invoke"
    assert dict(span.attributes)["output.value"] == "hello world"


def test_async_streaming_accumulates_output(spans):
    async def _drain():
        return [c async for c in FakeLLM().astream("go")]

    asyncio.run(_drain())
    (span,) = spans.get_finished_spans()
    assert span.name == "LLM Astream"
    assert dict(span.attributes)["output.value"] == "hello world"


def test_exception_marks_span_error_and_propagates(spans):
    with pytest.raises(ValueError, match="kaboom"):
        FakeLLM().boom("x")
    (span,) = spans.get_finished_spans()
    assert span.status.status_code == StatusCode.ERROR
    assert any(e.name == "exception" for e in span.events)


def test_session_id_is_attached_to_spans(spans):
    with T.session("conversation-42", user_id="user-7"):
        FakeLLM().invoke("hi")
    (span,) = spans.get_finished_spans()
    attrs = dict(span.attributes)
    assert attrs["session.id"] == "conversation-42"
    assert attrs["user.id"] == "user-7"


# --- session robustness (anti client-disconnect) --------------------------------


def test_session_cleanup_never_raises_across_contexts():
    """Reproduces the disconnect failure mode deterministically.

    A streaming async generator closed on client disconnect runs the session's
    cleanup in a *different* context than entry. With token-based reset that raised
    `ValueError: Token was created in a different Context`; snapshot/restore must not.
    """
    from gwenflow.telemetry.tracer import _telemetry_context

    cm = T.session("s1", user_id="u1")
    child_ctx = contextvars.copy_context()
    child_ctx.run(cm.__enter__)
    cm.__exit__(None, None, None)
    assert _telemetry_context.get() is None


def test_session_survives_generator_aclose():
    from gwenflow.telemetry.tracer import _telemetry_context

    async def stream():
        with T.session("thread-123"):
            for i in range(5):
                yield i

    async def run():
        gen = stream()
        assert await gen.__anext__() == 0
        assert _telemetry_context.get()["session.id"] == "thread-123"
        await gen.aclose()
        assert _telemetry_context.get() is None

    asyncio.run(run())


def test_session_nesting_restores_outer():
    from gwenflow.telemetry.tracer import _telemetry_context

    with T.session("outer", user_id="user-outer"):
        with T.session("inner", user_id="user-inner"):
            assert _telemetry_context.get()["session.id"] == "inner"
            assert _telemetry_context.get()["user.id"] == "user-inner"
        assert _telemetry_context.get()["session.id"] == "outer"
        assert _telemetry_context.get()["user.id"] == "user-outer"
    assert _telemetry_context.get() is None


# --- flexible context / metadata (project, environment, ...) --------------------


def test_context_is_pure_metadata(spans):
    with T.context(metadata={"session.id": "thread-1", "project.id": "acme", "project.name": "ACME Corp"}):
        FakeLLM().invoke("hi")
    attrs = dict(spans.get_finished_spans()[0].attributes)
    assert attrs["session.id"] == "thread-1"
    assert attrs["project.id"] == "acme"
    assert attrs["project.name"] == "ACME Corp"


def test_context_metadata_merges_when_nested(spans):
    with T.context(metadata={"project.id": "acme"}):
        with T.context(metadata={"session.id": "thread-9"}):
            FakeLLM().invoke("hi")
    attrs = dict(spans.get_finished_spans()[0].attributes)
    assert attrs["project.id"] == "acme"
    assert attrs["session.id"] == "thread-9"


def test_session_sugar_maps_to_openinference_keys(spans):
    with T.session("thread-1", user_id="user-1", metadata={"project.id": "acme"}):
        FakeLLM().invoke("hi")
    attrs = dict(spans.get_finished_spans()[0].attributes)
    assert attrs["session.id"] == "thread-1"
    assert attrs["user.id"] == "user-1"
    assert attrs["project.id"] == "acme"


# --- redaction ------------------------------------------------------------------


def test_capture_disabled_redacts_io(spans, monkeypatch):
    monkeypatch.setenv("GWENFLOW_TELEMETRY_CAPTURE_CONTENT", "false")
    FakeLLM().invoke("secret prompt")
    (span,) = spans.get_finished_spans()
    attrs = dict(span.attributes)
    assert attrs["input.value"] == "__REDACTED__"
    assert attrs["output.value"] == "__REDACTED__"
    assert attrs["llm.model_name"] == "gpt-test"
    assert attrs["llm.token_count.total"] == 20


def test_truncation(spans, monkeypatch):
    monkeypatch.setenv("GWENFLOW_TELEMETRY_MAX_ATTR_LENGTH", "10")
    FakeLLM().invoke("x" * 100)
    (span,) = spans.get_finished_spans()
    assert dict(span.attributes)["input.value"].startswith("xxxxxxxxxx... [truncated")
