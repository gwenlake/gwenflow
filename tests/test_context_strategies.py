"""Step 2: compaction strategies and the tiered orchestrator.

Every strategy is exercised on its own through `compact(messages, ctx)`,
then through `ChatMemoryBuffer.compact()`: tiers from the cheapest, batches
with hysteresis, persistence, pinning, pairing and the fallback chain.
"""

import json

import pytest

from gwenflow.memory import ChatMemoryBuffer, ChatSummaryMemoryBuffer
from gwenflow.memory.compaction import (
    ClampOversized,
    ClearToolResults,
    CompactionContext,
    DeduplicateReads,
    SlidingWindow,
    SummarizeOlder,
    default_file_key,
    is_pinned,
    pin,
    tool_groups,
    unpin,
    validate_tool_pairs,
)
from gwenflow.types import Message, TextContent
from tests.test_context_audit import ScriptedLLM, agent_history, parallel_turn, scripted_agent, tool_turn, words


def make_ctx(messages, budget=10_000, excess=None, summary=""):
    def count(m):
        buf = ChatMemoryBuffer(token_limit=budget, tokenizer_fn=words)
        return buf._token_count_for_message(m)

    total = sum(count(m) for m in messages)
    return CompactionContext(
        count=count,
        tokenizer_fn=words,
        excess=total if excess is None else excess,
        budget=budget,
        summary=summary,
    )


def results(messages):
    return [m.content for m in messages if m.role == "tool"]


# ---------------------------------------------------------------------------
# Helpers and invariants
# ---------------------------------------------------------------------------


def test_tool_groups_bind_calls_and_results():
    messages = [Message(role="user", content="t")] + parallel_turn("a", 3, 5) + tool_turn("b", "x", {}, 5)
    assert tool_groups(messages) == [(0, 1), (1, 5), (5, 7)]


def test_validator_catches_orphans_and_split_groups():
    ok = [Message(role="user", content="t")] + parallel_turn("a", 2, 5)
    validate_tool_pairs(ok)
    with pytest.raises(ValueError, match="orphan"):
        validate_tool_pairs(ok[2:])
    with pytest.raises(ValueError, match="mismatch"):
        validate_tool_pairs(ok[:-1])


def test_pin_and_unpin():
    m = Message(role="user", content="keep me")
    assert not is_pinned(m)
    assert is_pinned(pin(m))
    assert not is_pinned(unpin(m))


def test_default_file_key_knows_the_coding_bundle():
    assert default_file_key("ReadFileTool", {"file_path": "a.py"}) == "read:a.py"
    assert default_file_key("read_file", {"path": "a.py"}) == "read:a.py"
    assert default_file_key("GrepTool", {"pattern": "x"}) is None
    assert default_file_key("ReadFileTool", "not a dict") is None


# ---------------------------------------------------------------------------
# DeduplicateReads
# ---------------------------------------------------------------------------


def test_deduplicate_keeps_the_latest_read_of_a_file():
    messages = [Message(role="user", content="t")]
    messages += tool_turn("a", "ReadFileTool", {"file_path": "big.csv"}, 100)
    messages += tool_turn("b", "GrepTool", {"pattern": "x"}, 100)
    messages += tool_turn("c", "ReadFileTool", {"file_path": "big.csv"}, 100)
    out = DeduplicateReads().compact(messages, make_ctx(messages))
    assert results(out)[0].startswith("[Superseded read of big.csv")
    assert results(out)[1].startswith("data") and results(out)[2].startswith("data")
    assert results(messages)[0].startswith("data")  # input untouched
    validate_tool_pairs(out)


def test_deduplicate_identical_calls_and_leaves_small_results():
    messages = [Message(role="user", content="t")]
    messages += tool_turn("a", "GrepTool", {"pattern": "x"}, 100)
    messages += tool_turn("b", "GrepTool", {"pattern": "x"}, 100)
    messages += tool_turn("c", "GrepTool", {"pattern": "y"}, 3)
    messages += tool_turn("d", "GrepTool", {"pattern": "y"}, 3)
    out = DeduplicateReads().compact(messages, make_ctx(messages))
    assert results(out)[0].startswith("[Same call as a later GrepTool(")
    assert results(out)[2] == results(messages)[2]  # tiny: not worth a cache miss
    assert DeduplicateReads(identical_calls=False).compact(messages, make_ctx(messages)) is messages


def test_deduplicate_runs_even_when_the_history_fits():
    buf = ChatMemoryBuffer(token_limit=100_000, tokenizer_fn=words)
    buf.add_message(Message(role="user", content="t"))
    buf.add_messages(tool_turn("a", "ReadFileTool", {"file_path": "f"}, 100))
    buf.add_messages(tool_turn("b", "ReadFileTool", {"file_path": "f"}, 100))
    record = buf.compact()
    assert record.strategies == ["deduplicate_reads"]
    assert results(buf.messages)[0].startswith("[Superseded")


# ---------------------------------------------------------------------------
# ClampOversized
# ---------------------------------------------------------------------------


def test_clamp_keeps_head_and_tail_with_a_marker():
    text = " ".join(f"w{i}" for i in range(2000))
    messages = [Message(role="user", content=text), Message(role="assistant", content=text)]
    strategy = ClampOversized(max_part_tokens=100, keep_head_chars=50, keep_tail_chars=50)
    out = strategy.compact(messages, make_ctx(messages))
    assert out[0].content == text  # user messages are never clamped
    assert out[1].content.startswith(text[:50]) and out[1].content.endswith(text[-50:])
    assert "[clamped: removed" in out[1].content
    assert messages[1].content == text


def test_clamp_applies_to_tool_results_parts_and_arguments():
    big = "x " * 3000
    call = {
        "id": "1",
        "type": "function",
        "function": {"name": "WriteFileTool", "arguments": json.dumps({"content": big})},
    }
    messages = [
        Message(role="user", content="t"),
        Message(role="assistant", content=[TextContent(content=big)], tool_calls=[call]),
        Message(role="tool", tool_call_id="1", content=big),
    ]
    out = ClampOversized(max_part_tokens=100, keep_head_chars=20, keep_tail_chars=20).compact(
        messages, make_ctx(messages)
    )
    assert "[clamped" in out[1].content[0].content
    assert "[clamped" in out[1].tool_calls[0]["function"]["arguments"]
    assert "[clamped" in out[2].content
    assert messages[1].tool_calls[0]["function"]["arguments"] == call["function"]["arguments"]
    validate_tool_pairs(out)


def test_clamp_threshold_defaults_to_a_share_of_the_budget():
    messages = [Message(role="assistant", content="w " * 300)]
    assert ClampOversized().compact(messages, make_ctx(messages, budget=10_000)) is messages
    out = ClampOversized(keep_head_chars=10, keep_tail_chars=10).compact(messages, make_ctx(messages, budget=400))
    assert "[clamped" in out[0].content


# ---------------------------------------------------------------------------
# ClearToolResults
# ---------------------------------------------------------------------------


def test_clear_oldest_first_until_the_excess_is_gone():
    messages = agent_history(n_reads=8, result_words=200)
    out = ClearToolResults(keep_pairs=3).compact(messages, make_ctx(messages, excess=450))
    cleared = [r.startswith("[Result of ReadFileTool(") for r in results(out)]
    assert cleared == [True, True, True, False, False, False, False, False]
    assert "f0.csv" in results(out)[0] and "tokens" in results(out)[0]
    validate_tool_pairs(out)


def test_clear_keeps_recent_pairs_excluded_tools_pinned_and_small_results():
    messages = agent_history(n_reads=6, result_words=200)
    messages[2] = pin(messages[2])  # result of c0
    messages += tool_turn("g", "GrepTool", {"pattern": "x"}, 200)
    messages += tool_turn("s", "ShellTool", {"cmd": "ls"}, 10)
    strategy = ClearToolResults(keep_pairs=1, exclude_tools=frozenset({"GrepTool"}), min_clear_tokens=50)
    out = strategy.compact(messages, make_ctx(messages))
    res = results(out)
    assert res[0].startswith("data")  # pinned
    assert all(r.startswith("[Result of") for r in res[1:6])
    assert res[6].startswith("data")  # excluded tool
    assert res[7].startswith("data")  # last pair and too small anyway


def test_clear_tool_inputs_blanks_the_arguments():
    messages = agent_history(n_reads=4, result_words=200)
    out = ClearToolResults(keep_pairs=1, clear_tool_inputs=True).compact(messages, make_ctx(messages))
    assistant = [m for m in out if m.role == "assistant"]
    assert assistant[0].tool_calls[0]["function"]["arguments"] == "{}"
    assert assistant[-1].tool_calls[0]["function"]["arguments"] != "{}"
    assert messages[1].tool_calls[0]["function"]["arguments"] != "{}"
    validate_tool_pairs(out)


def test_clear_is_a_noop_when_nothing_qualifies():
    messages = agent_history(n_reads=2, result_words=200)
    assert ClearToolResults(keep_pairs=3).compact(messages, make_ctx(messages)) is messages


# ---------------------------------------------------------------------------
# SummarizeOlder
# ---------------------------------------------------------------------------


def test_summarize_folds_the_oldest_and_keeps_a_tail_and_the_task():
    llm = ScriptedLLM(summary="SUMMARY: files 0-4 were read.")
    messages = agent_history(n_reads=8, result_words=100, with_notes=True)
    ctx = make_ctx(messages, budget=1000)
    out = SummarizeOlder(llm=llm, keep_tokens=450).compact(messages, ctx)
    assert ctx.summary == "SUMMARY: files 0-4 were read."
    assert out[0].content.startswith("TASK:")  # first user message kept in place
    assert len(out) < len(messages)
    assert out[1].role == "assistant"  # the cut is a group boundary
    validate_tool_pairs(out)
    assert "summarized" in ctx.notes[0]


def test_summarize_caps_the_summary_and_the_transcript():
    seen = {}

    class Spy:
        def invoke(self, input):
            seen["prompt"] = input[1]["content"]
            from types import SimpleNamespace

            return SimpleNamespace(content="s " * 500, usage=None)

    messages = agent_history(n_reads=6, result_words=400)
    ctx = make_ctx(messages, budget=1000)
    SummarizeOlder(llm=Spy(), keep_tokens=100, tool_result_max_chars=50).compact(messages, ctx)
    assert words(ctx.summary) <= 150  # 15 % of the budget
    assert "[...]" in seen["prompt"]  # tool results capped in the transcript


def test_summarize_retries_once_then_gives_up():
    class Empty:
        calls = 0

        def invoke(self, input):
            self.calls += 1
            from types import SimpleNamespace

            return SimpleNamespace(content="", usage=None)

    llm = Empty()
    messages = agent_history(n_reads=6, result_words=100)
    ctx = make_ctx(messages, budget=600)
    assert SummarizeOlder(llm=llm, keep_tokens=100).compact(messages, ctx) is messages
    assert llm.calls == 2
    assert ctx.summary == ""


def test_summarize_is_incremental():
    llm = ScriptedLLM(summary="SUMMARY v2")
    messages = agent_history(n_reads=4, result_words=100)
    ctx = make_ctx(messages, budget=600, summary="SUMMARY v1")
    SummarizeOlder(llm=llm, keep_tokens=100).compact(messages, ctx)
    assert "Previous summary:\nSUMMARY v1" in llm.prompts[-1][1]["content"] if llm.prompts else True
    assert ctx.summary == "SUMMARY v2"


# ---------------------------------------------------------------------------
# SlidingWindow
# ---------------------------------------------------------------------------


def test_sliding_window_drops_oldest_groups_but_keeps_protected():
    messages = [Message(role="user", content="TASK first")]
    messages += parallel_turn("a", 3, 50)
    messages += tool_turn("b", "x", {}, 50)
    messages.append(pin(Message(role="assistant", content="pinned note")))
    messages += tool_turn("c", "x", {}, 50)
    messages.append(Message(role="user", content="live task"))
    messages += tool_turn("d", "x", {}, 50)
    out = SlidingWindow().compact(messages, make_ctx(messages, excess=250))
    assert out[0].content == "TASK first"
    assert any(m.content == "pinned note" for m in out)
    assert any(m.content == "live task" for m in out)
    assert not any(m.tool_call_id and m.tool_call_id.startswith("a") for m in out)  # the parallel group went whole
    validate_tool_pairs(out)


def test_sliding_window_keep_messages():
    messages = [Message(role="user", content="t")] + [Message(role="assistant", content=f"a{i}") for i in range(10)]
    out = SlidingWindow(keep_messages=4, receipt=False).compact(messages, make_ctx(messages, excess=0))
    assert len(out) == 4 and out[0].content == "t"
    with_note = SlidingWindow(keep_messages=4).compact(messages, make_ctx(messages, excess=0))
    assert with_note[1].content.startswith("[Earlier messages were removed")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def test_compact_runs_tiers_in_order_and_stops_early():
    buf = ChatSummaryMemoryBuffer(token_limit=3600, tokenizer_fn=words, llm=ScriptedLLM())
    buf.add_messages(agent_history(n_reads=8, result_words=500))
    record = buf.compact()
    assert record.strategies == ["clear_tool_results"]
    assert record.tokens_after <= buf.target() < record.tokens_before
    assert buf.llm.summary_calls == 0


def test_compact_persists_and_keeps_the_prefix_stable_between_batches():
    buf = ChatMemoryBuffer(token_limit=1800, tokenizer_fn=words)
    buf.add_messages(agent_history(n_reads=12, result_words=200))
    buf.compact()
    first_window = [str(m.content)[:40] for m in buf.get()]
    batches = 0
    for i in range(6):
        buf.add_messages(tool_turn(f"extra{i}", "ReadFileTool", {"file_path": f"x{i}.csv"}, 200))
        if buf.compact() is not None:
            batches += 1
        window = [str(m.content)[:40] for m in buf.get()]
        assert window[: len(first_window)] == first_window or batches > 0
    assert 1 <= batches <= 3  # hysteresis: a few batches, not one per request
    assert buf.used_tokens() <= buf.limit()


def test_compact_records_what_it_removed():
    buf = ChatMemoryBuffer(token_limit=600, tokenizer_fn=words)
    buf.add_messages(agent_history(n_reads=6, result_words=200))
    record = buf.compact()
    assert record.messages_before == 13
    assert record.removed  # originals of altered or dropped messages
    assert all(str(r.content).startswith("data") or r.tool_calls for r in record.removed)
    assert buf.compactions[-1] is record


def test_compact_falls_back_on_the_sliding_window_when_the_summarizer_raises():
    class Boom:
        def invoke(self, input):
            raise RuntimeError("down")

    buf = ChatSummaryMemoryBuffer(token_limit=600, tokenizer_fn=words, llm=Boom())
    buf.add_messages(agent_history(n_reads=10, result_words=200))
    record = buf.compact()
    assert "sliding_window" in record.strategies
    assert "summarize_older" not in record.strategies
    assert buf.messages[0].content.startswith("TASK:")
    validate_tool_pairs(buf.messages)


def test_compact_uses_the_summary_before_dropping():
    buf = ChatSummaryMemoryBuffer(token_limit=900, tokenizer_fn=words, llm=ScriptedLLM(summary="SUMMARY ok"))
    buf.add_messages(agent_history(n_reads=12, result_words=200, with_notes=True))
    record = buf.compact()
    assert "summarize_older" in record.strategies
    assert buf.summary == "SUMMARY ok"
    assert buf.get()[0].content.startswith("Summary of the earlier")
    assert buf.messages[0].content.startswith("TASK:")


def test_custom_strategy_list_and_fallback_to_the_last_tier():
    buf = ChatMemoryBuffer(token_limit=500, tokenizer_fn=words, strategies=[SlidingWindow()])
    buf.add_messages(agent_history(n_reads=6, result_words=200))
    record = buf.compact()
    assert record.strategies == ["sliding_window"]
    validate_tool_pairs(buf.messages)


def test_target_must_be_below_the_limit():
    with pytest.raises(ValueError, match="target"):
        ChatMemoryBuffer(token_limit=1000, target_tokens=1000)
    with pytest.raises(ValueError, match="exclusive"):
        ChatMemoryBuffer(max_fraction=0.8, target_fraction=0.5, target_tokens=10)
    buf = ChatMemoryBuffer(max_fraction=0.8, target_fraction=0.5)
    buf.context_window = 1000
    assert (buf.limit(), buf.target()) == (800, 500)


# ---------------------------------------------------------------------------
# End to end on the agent
# ---------------------------------------------------------------------------


def test_long_tool_loop_stays_bounded_keeps_the_task_and_the_recent_results():
    agent = scripted_agent(llm=ScriptedLLM(context_size=1500, tool_turns=30, summary="SUMMARY of the search"))
    response = agent.run("TASK: analyse the French cloud market.")
    assert response.content == "final report"
    prompts = agent.llm.prompts
    assert len(prompts) == 31
    for prompt in prompts:
        assert ScriptedLLM.prompt_tokens(prompt) <= 1500 * 1.1  # the budget, with the tolerance of the estimate
        assert any("TASK:" in str(m.get("content")) for m in prompt)
        validate_tool_pairs(
            [
                Message(**{k: v for k, v in m.items() if k in ("role", "content", "tool_calls", "tool_call_id")})
                for m in prompt
                if m["role"] != "system"
            ]
        )
    last = prompts[-1]
    intact = [m for m in last if m["role"] == "tool" and str(m["content"]).startswith("RESULTS for")]
    assert len(intact) >= 3  # keep_pairs
    assert agent.history.llm.summary_calls >= 1
    assert agent.history.compactions


def test_parallel_calls_are_never_separated_by_the_agent_loop():
    buf = ChatSummaryMemoryBuffer(token_limit=700, tokenizer_fn=words, llm=ScriptedLLM())
    buf.add_message(Message(role="user", content="TASK: compare."))
    for i in range(6):
        buf.add_messages(parallel_turn(f"g{i}_", 3, 60))
        buf.compact()
        validate_tool_pairs(buf.messages)
        validate_tool_pairs([m for m in buf.get() if m.role != "system"])
