"""Step 4: observability.

Context usage exposed per request, limit notices in the system prompt of the
request only, and a `compact_messages` span with counts, never content.
"""

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from gwenflow.memory import ChatSummaryMemoryBuffer
from gwenflow.telemetry import tracer
from gwenflow.telemetry._settings import set_tracing_enabled
from gwenflow.types import ContextUsage
from tests.test_context_audit import ScriptedLLM, agent_history, scripted_agent, words


@pytest.fixture
def spans():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer._tracer = provider.get_tracer("test")
    set_tracing_enabled(True)
    try:
        yield exporter
    finally:
        set_tracing_enabled(False)
        tracer._tracer = None
        exporter.clear()


def system_prompt_of(prompt) -> str:
    return prompt[0]["content"] if prompt and prompt[0]["role"] == "system" else ""


# ---------------------------------------------------------------------------
# Usage report
# ---------------------------------------------------------------------------


def test_on_context_usage_is_called_before_every_request():
    seen: list[ContextUsage] = []
    agent = scripted_agent(llm=ScriptedLLM(context_size=5000, tool_turns=3), on_context_usage=seen.append)
    agent.run("TASK: go.")
    assert len(seen) == 4 == len(agent.llm.prompts)
    assert seen[0].anchored is False and seen[1].anchored is True
    assert all(u.window_tokens > 0 and u.limit_tokens < u.window_tokens for u in seen)
    assert seen[-1].used_tokens > seen[0].used_tokens
    assert 0 < seen[-1].fraction < 1


def test_context_usage_method_reports_the_current_state():
    agent = scripted_agent(llm=ScriptedLLM(context_size=5000, tool_turns=1))
    agent.run("TASK: go.")
    usage = agent.context_usage()
    assert usage.resolved is True
    assert usage.used_tokens == agent.history.used_tokens()


# ---------------------------------------------------------------------------
# Limit notices
# ---------------------------------------------------------------------------


def test_no_notice_by_default():
    agent = scripted_agent(llm=ScriptedLLM(context_size=400, tool_turns=6, result_words=150))
    agent.run("TASK: go.")
    assert not any("Context notice" in system_prompt_of(p) for p in agent.llm.prompts)
    assert "Context notice" not in agent.history.system_prompt


def test_context_notice_escalates_and_is_request_scoped():
    agent = scripted_agent(llm=ScriptedLLM(context_size=600, tool_turns=8, result_words=120), warn_near_limits=True)
    agent.run("TASK: go.")
    prompts = [system_prompt_of(p) for p in agent.llm.prompts]
    assert "Context notice" not in prompts[0]  # nothing to say on the first request
    assert any("[URGENT]" in p and "context window" in p for p in prompts)
    assert any("[CRITICAL]" in p and "compacted" in p for p in prompts)
    # the notice is never stored: no message carries it, and the next run starts clean
    assert not any("Context notice" in str(m.content) for m in agent.history.messages)
    agent.run("TASK: again, briefly.")
    assert "Context notice" not in system_prompt_of(agent.llm.prompts[-1]) or agent.llm.prompts[-1]


def test_turn_notice_counts_down_to_critical():
    agent = scripted_agent(llm=ScriptedLLM(context_size=50_000, tool_turns=20), warn_near_limits=True, max_turns=6)
    agent.run("TASK: go.")
    prompts = [system_prompt_of(p) for p in agent.llm.prompts]
    assert len(prompts) == 6
    assert "Context notice" not in prompts[0]
    assert "[CRITICAL] 2 turn(s) left" in prompts[-3] or "[CRITICAL] 3 turn(s) left" in prompts[-3]
    assert "[CRITICAL] 0 turn(s) left" in prompts[-1]


def test_notice_thresholds_are_configurable():
    agent = scripted_agent(
        llm=ScriptedLLM(context_size=50_000, tool_turns=20),
        warn_near_limits=True,
        max_turns=10,
        warning_threshold=0.5,
        critical_remaining_turns=1,
    )
    agent.run("TASK: go.")
    prompts = [system_prompt_of(p) for p in agent.llm.prompts]
    urgent = [i for i, p in enumerate(prompts) if "[URGENT]" in p]
    critical = [i for i, p in enumerate(prompts) if "[CRITICAL]" in p]
    assert urgent and urgent[0] == 4  # the 5th request: 50 % of 10 turns used, this one included
    assert critical == [8, 9]  # 1 turn left after this one, then 0


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------


def test_compact_messages_span_carries_counts_but_no_content(spans):
    buf = ChatSummaryMemoryBuffer(token_limit=900, tokenizer_fn=words, llm=ScriptedLLM(summary="SECRET SUMMARY"))
    buf.add_messages(agent_history(n_reads=12, result_words=200, with_notes=True))
    record = buf.compact()
    assert record is not None
    names = [s.name for s in spans.get_finished_spans()]
    assert "compact_messages" in names
    span = next(s for s in spans.get_finished_spans() if s.name == "compact_messages")
    attrs = dict(span.attributes)
    assert list(attrs["gwenflow.compaction.strategies"]) == record.strategies
    assert attrs["gwenflow.compaction.messages_before"] == record.messages_before
    assert attrs["gwenflow.compaction.messages_after"] == record.messages_after
    assert attrs["gwenflow.compaction.tokens_before"] > attrs["gwenflow.compaction.tokens_after"]
    assert "summarized" in " ".join(attrs["gwenflow.compaction.notes"])
    rendered = str(attrs)
    assert "SECRET" not in rendered and "data data" not in rendered and "NOTE 0" not in rendered


def test_no_span_when_the_history_fits(spans):
    buf = ChatSummaryMemoryBuffer(token_limit=100_000, tokenizer_fn=words, llm=ScriptedLLM())
    buf.add_messages(agent_history(n_reads=3, result_words=20))
    assert buf.compact() is None
    assert [s.name for s in spans.get_finished_spans()] == []


def test_span_is_a_noop_when_tracing_is_off():
    with tracer.span("compact_messages") as span:
        span.set_attribute("x", 1)  # must not raise
