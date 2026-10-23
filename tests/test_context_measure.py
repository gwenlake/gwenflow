"""Step 1: measuring the context on the provider's counts.

Usage anchoring, `max_fraction` against the window of the model actually
called, `fallback_context_window`, and an injectable tokenizer.
"""

import sys
import types

import pytest

from gwenflow import Agent, ChatOpenAI
from gwenflow.llms.base import DEFAULT_CONTEXT_SIZE
from gwenflow.memory import ChatMemoryBuffer, ChatSummaryMemoryBuffer
from gwenflow.memory.chat_memory_buffer import DEFAULT_CONTEXT_WINDOW, DEFAULT_MAX_FRACTION
from gwenflow.types import ContextWindowInfo, Message
from gwenflow.utils.tokens import hf_tokenizer
from tests.test_context_audit import ScriptedLLM, agent_history, scripted_agent, tool_turn, words

# ---------------------------------------------------------------------------
# Window resolution on the LLM
# ---------------------------------------------------------------------------


def test_context_window_info_reports_where_the_value_comes_from():
    assert ChatOpenAI(model="gpt-5-mini", api_key="x").context_window_info() == ContextWindowInfo(
        tokens=400000, resolved=True, source="table"
    )
    assert ChatOpenAI(model="x", api_key="x", context_window=32768).context_window_info() == ContextWindowInfo(
        tokens=32768, resolved=True, source="context_window"
    )
    assert ChatOpenAI(model="x", api_key="x", fallback_context_window=32768).context_window_info() == ContextWindowInfo(
        tokens=32768, resolved=False, source="fallback"
    )
    assert ChatOpenAI(model="x", api_key="x").context_window_info() == ContextWindowInfo(
        tokens=DEFAULT_CONTEXT_SIZE, resolved=False, source="default"
    )


def test_context_window_wins_over_fallback_and_table():
    llm = ChatOpenAI(model="gpt-5-mini", api_key="x", context_window=1000, fallback_context_window=2000)
    assert llm.resolve_context_window() == 1000


# ---------------------------------------------------------------------------
# max_fraction / token_limit on the buffer
# ---------------------------------------------------------------------------


def test_token_limit_and_max_fraction_are_exclusive():
    with pytest.raises(ValueError, match="exclusive"):
        ChatMemoryBuffer(token_limit=1000, max_fraction=0.8)
    with pytest.raises(ValueError, match="max_fraction"):
        ChatMemoryBuffer(max_fraction=1.5)


def test_max_fraction_resolves_against_the_current_window():
    buf = ChatMemoryBuffer(max_fraction=0.5, tokenizer_fn=words)
    assert buf.limit() == int(DEFAULT_CONTEXT_WINDOW * 0.5)  # nothing known yet
    buf.context_window = 32768
    assert buf.limit() == 16384
    assert buf.token_limit is None


def test_token_limit_is_absolute():
    buf = ChatMemoryBuffer(token_limit=1234, tokenizer_fn=words)
    buf.context_window = 1_000_000
    assert buf.limit() == 1234


def test_agent_default_history_follows_the_llm_window():
    agent = Agent(name="a", llm=ChatOpenAI(model="gpt-5-mini", api_key="x"))
    assert agent.history.max_fraction == DEFAULT_MAX_FRACTION
    assert agent.history.limit() == int(400000 * DEFAULT_MAX_FRACTION)
    agent.llm = ChatOpenAI(model="my-vllm-model", api_key="x", context_window=32768)
    agent._prepare_history(task="t")
    assert agent.history.limit() == int(32768 * DEFAULT_MAX_FRACTION)
    assert agent.history.context_window_resolved is True


def test_agent_context_usage_flags_an_assumed_window():
    agent = Agent(name="a", llm=ChatOpenAI(model="my-vllm-model", api_key="x", fallback_context_window=32768))
    usage = agent.context_usage()
    assert usage.window_tokens == 32768
    assert usage.resolved is False
    assert usage.limit_tokens == int(32768 * DEFAULT_MAX_FRACTION)


def test_clone_keeps_the_fraction_configuration():
    agent = Agent(name="a", llm=ChatOpenAI(model="gpt-5-mini", api_key="x"))
    clone = agent.clone()
    assert clone.history.max_fraction == agent.history.max_fraction
    assert clone.history.limit() == agent.history.limit()


# ---------------------------------------------------------------------------
# Usage anchoring
# ---------------------------------------------------------------------------


def test_unanchored_usage_is_a_scaled_local_estimate():
    buf = ChatMemoryBuffer(token_limit=10_000, tokenizer_fn=words)
    buf.system_prompt = "a b c d"
    buf.reserved_tokens = 50
    buf.add_message(Message(role="user", content="one two three"))
    local = buf._initial_token_count() + 50 + buf._token_count_for_messages(buf.messages)
    assert buf.used_tokens() == local
    assert buf.context_usage().anchored is False
    buf.token_scale = 2.0
    assert buf.used_tokens() == 2 * local


def test_anchored_usage_adds_only_the_messages_stored_since():
    buf = ChatMemoryBuffer(token_limit=10_000, tokenizer_fn=words)
    buf.add_message(Message(role="user", content="word " * 300))
    window = buf.get()
    buf.add_message(Message(role="assistant", content="the reply"))
    buf.observe_usage(window, input_tokens=5000, output_tokens=20)  # the provider's view
    assert buf.used_tokens() == 5020
    assert buf.context_usage().anchored is True
    buf.add_message(Message(role="tool", tool_call_id="x", content="one two three four"))
    assert buf.used_tokens() == 5020 + buf._scaled(buf._token_count_for_message(buf.messages[-1]))


def test_anchor_corrects_an_underestimating_tokenizer():
    """The provider billed far more than the local count: the anchor wins for what was sent."""
    buf = ChatMemoryBuffer(token_limit=10_000, tokenizer_fn=words)
    buf.add_message(Message(role="user", content="word " * 300))
    local_before = buf.used_tokens()
    window = buf.get()
    buf.add_message(Message(role="assistant", content="ok"))
    buf.observe_usage(window, input_tokens=local_before * 3, output_tokens=1)
    assert buf.used_tokens() >= local_before * 3
    assert buf.token_scale > 2.5  # and the scale follows for the next estimates


def test_anchor_is_dropped_when_the_history_changes_shape():
    buf = ChatMemoryBuffer(token_limit=10_000, tokenizer_fn=words)
    buf.add_message(Message(role="user", content="word " * 300))
    buf.observe_usage(buf.get(), input_tokens=5000)
    assert buf.context_usage().anchored
    buf.reset()
    assert not buf.context_usage().anchored
    buf.add_message(Message(role="user", content="hi"))
    # back to the local estimate; the calibrated scale is a property of the tokenizer and stays
    assert buf.used_tokens() == buf._scaled(buf._token_count_for_messages(buf.messages))


def test_summary_compaction_drops_the_anchor():
    buf = ChatSummaryMemoryBuffer(token_limit=200, tokenizer_fn=words, llm=ScriptedLLM())
    buf.add_messages([Message(role="user", content="q " * 80), Message(role="assistant", content="a " * 80)])
    buf.add_messages(agent_history(n_reads=3, result_words=40))
    buf.observe_usage(buf.get(), input_tokens=900)
    buf.compact()
    assert buf.summary != ""
    assert not buf.context_usage().anchored


def test_agent_anchors_after_each_reply():
    agent = scripted_agent(llm=ScriptedLLM(context_size=5000, tool_turns=2))
    agent.run("TASK: go.")
    usage = agent.history.context_usage()
    assert usage.anchored is True
    # the last reply is the final answer: the anchor covers the whole history
    assert usage.used_tokens == ScriptedLLM.prompt_tokens(agent.llm.prompts[-1]) + 10
    assert agent.history._anchor_index == len(agent.history.messages)


# ---------------------------------------------------------------------------
# Injectable tokenizer
# ---------------------------------------------------------------------------


def test_hf_tokenizer_wraps_a_transformers_tokenizer(monkeypatch):
    class FakeTok:
        def encode(self, text, add_special_tokens=True):
            assert add_special_tokens is False
            return list(text)  # one token per character

    fake = types.ModuleType("transformers")
    fake.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda name, **kw: FakeTok())
    monkeypatch.setitem(sys.modules, "transformers", fake)
    count = hf_tokenizer("Qwen/Qwen3.6-27B")
    assert count("abcd") == 4
    buf = ChatMemoryBuffer(token_limit=100, tokenizer_fn=count)
    assert buf._token_count_for_message(Message(role="user", content="abcd")) == 4 + 4


def test_hf_tokenizer_without_transformers_is_explicit(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(ImportError, match="transformers"):
        hf_tokenizer("whatever")


def test_tool_turn_helper_is_paired():
    assert [m.role for m in tool_turn("c", "t", {}, 3)] == ["assistant", "tool"]
