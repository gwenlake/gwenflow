"""Audit of the context management (step 0).

Each test either confirms a behaviour that is already right, or demonstrates
a defect. Defects are marked `xfail(strict=True)`: the test states the
*wanted* behaviour, fails today, and the marker is removed by the step that
fixes it. Everything runs offline, with a word-count tokenizer.
"""

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from gwenflow.agents.agent import Agent
from gwenflow.llms import ChatOpenAI
from gwenflow.llms.base import CONTEXT_SAFETY_RATIO, DEFAULT_CONTEXT_SIZE
from gwenflow.memory import ChatMemoryBuffer, ChatSummaryMemoryBuffer
from gwenflow.types import Message, ModelResponse, RequestUsage, TextContent, ToolCall

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def words(text) -> int:
    """One token per word: deterministic, readable budgets."""
    return len(str(text).split())


def tool_turn(call_id: str, name: str, arguments: dict, result_words: int, note: str = "") -> list[Message]:
    call = {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}
    return [
        Message(role="assistant", content=note or None, tool_calls=[call]),
        Message(role="tool", tool_call_id=call_id, content=" ".join(["data"] * result_words)),
    ]


def parallel_turn(prefix: str, n: int, result_words: int) -> list[Message]:
    """One assistant message with `n` parallel calls, then their `n` results."""
    calls = [
        {"id": f"{prefix}{k}", "type": "function", "function": {"name": "search", "arguments": json.dumps({"q": k})}}
        for k in range(n)
    ]
    results = [
        Message(role="tool", tool_call_id=f"{prefix}{k}", content=" ".join(["data"] * result_words)) for k in range(n)
    ]
    return [Message(role="assistant", content=None, tool_calls=calls)] + results


def agent_history(n_reads: int = 20, result_words: int = 200, with_notes: bool = False) -> list[Message]:
    messages = [Message(role="user", content="TASK: analyse every file and write a report.")]
    for i in range(n_reads):
        note = f"NOTE {i}: file f{i}.csv looks relevant because of column {i}." if with_notes else ""
        messages += tool_turn(f"c{i}", "ReadFileTool", {"file_path": f"f{i}.csv"}, result_words, note=note)
    return messages


def assert_tool_pairs(messages: list[Message]) -> None:
    """Invariant every provider enforces: calls and results come in matched groups.

    Each assistant message carrying tool_calls must be immediately followed by
    exactly one `tool` message per call id (any order), and a `tool` message
    may only appear in such a group.
    """
    index = 0
    while index < len(messages):
        message = messages[index]
        if message.role == "tool":
            raise AssertionError(f"orphan tool result at {index}: {message.tool_call_id}")
        if message.role == "assistant" and message.tool_calls:
            expected = {call["id"] for call in message.tool_calls}
            index += 1
            seen: set[str] = set()
            while index < len(messages) and messages[index].role == "tool":
                seen.add(messages[index].tool_call_id)
                index += 1
            if seen != expected:
                raise AssertionError(f"tool group mismatch: calls {sorted(expected)} results {sorted(seen)}")
            continue
        index += 1


@dataclass
class ScriptedLLM:
    """Fake LLM for the agent loop: `tool_turns` tool calls, then a final answer.

    Doubles as the summarizer (the compaction prompt is recognised) so that a
    single instance covers everything `Agent` needs. Records every prompt it
    receives in `prompts` and reports a fixed prompt-token usage.
    """

    context_size: int = 600
    tool_turns: int = 6
    result_words: int = 150
    summary: str = "SUMMARY: the user asked for a report."
    tools: list = field(default_factory=list)
    tool_choice: Any = None
    response_format: Any = None
    model: str = "scripted"
    prompts: list = field(default_factory=list)
    summary_calls: int = 0
    turn: int = 0

    def get_context_size(self):
        return self.context_size

    def input_to_message_list(self, input):
        if isinstance(input, str):
            return [Message(role="user", content=input)]
        return [m if isinstance(m, Message) else Message(**m) for m in input]

    def get_thinking_parts(self, response):
        return None

    @staticmethod
    def prompt_tokens(input) -> int:
        """Realistic usage for the fakes: one token per word, like `words()`."""
        return sum(len(str(m.get("content") or "").split()) + 4 for m in input)

    def invoke(self, input):
        first = input[0].get("content", "") if input and isinstance(input[0], dict) else ""
        if first.startswith("You maintain a running summary"):
            self.summary_calls += 1
            return ModelResponse(
                parts=[TextContent(content=self.summary)],
                usage=RequestUsage(input_tokens=self.prompt_tokens(input), output_tokens=20),
            )
        self.prompts.append(input)
        self.turn += 1
        usage = RequestUsage(input_tokens=self.prompt_tokens(input), output_tokens=10)
        if self.turn <= self.tool_turns:
            call = ToolCall(id=f"call_{self.turn}", name="search", arguments=json.dumps({"q": f"market {self.turn}"}))
            return ModelResponse(parts=[call], finish_reason="tool_calls", usage=usage)
        return ModelResponse(parts=[TextContent(content="final report")], finish_reason="stop", usage=usage)

    async def ainvoke(self, input):
        return self.invoke(input)


def search(q: str) -> str:
    """Search the web.

    Args:
        q: what to look for.
    """
    return f"RESULTS for {q}: " + "lorem ipsum " * 75


def scripted_agent(**kwargs) -> Agent:
    from gwenflow.tools import Tool

    llm = kwargs.pop("llm", None) or ScriptedLLM()
    agent = Agent(name="auditor", instructions="Be terse.", llm=llm, tools=[Tool(search)], **kwargs)
    agent.history.tokenizer_fn = words
    return agent


# ---------------------------------------------------------------------------
# 1. Context window resolution
# ---------------------------------------------------------------------------


def test_known_and_served_model_names_resolve_to_the_table():
    """Confirmed: the table, tolerant names, and the explicit override all work."""
    assert ChatOpenAI(model="gpt-5-mini", api_key="x").get_context_size() == int(400000 * CONTEXT_SAFETY_RATIO)
    assert ChatOpenAI(model="Qwen/Qwen3.6-27B-AWQ", api_key="x").get_context_size() == int(
        262144 * CONTEXT_SAFETY_RATIO
    )
    assert ChatOpenAI(model="anything", api_key="x", context_window=32768).get_context_size() == int(
        32768 * CONTEXT_SAFETY_RATIO
    )


def test_unknown_model_silently_gets_the_128k_default():
    """Confirmed (and documented): an unknown model is assumed to have 128k."""
    llm = ChatOpenAI(model="my-local-model", api_key="x")
    assert llm.get_context_size() == int(DEFAULT_CONTEXT_SIZE * CONTEXT_SAFETY_RATIO)


def test_fallback_context_window_applies_only_to_unknown_models():
    """Fixed in step 1."""
    known = ChatOpenAI(model="gpt-5-mini", api_key="x", fallback_context_window=32768)
    unknown = ChatOpenAI(model="my-local-model", api_key="x", fallback_context_window=32768)
    assert known.get_context_size() == int(400000 * CONTEXT_SAFETY_RATIO)
    assert unknown.get_context_size() == int(32768 * CONTEXT_SAFETY_RATIO)


def test_budget_follows_a_replaced_llm():
    """`agent.llm = other` must change the window the history works with (fixed in step 1)."""
    agent = scripted_agent(llm=ScriptedLLM(context_size=600))
    assert agent.history.limit() == 600
    agent.llm = ScriptedLLM(context_size=200)
    agent.run("TASK: analyse the market.")
    assert agent.history.limit() == 200


# ---------------------------------------------------------------------------
# 2. What the budget counts
# ---------------------------------------------------------------------------


def test_budget_counts_system_prompt_tool_schemas_and_summary():
    """Confirmed: all three are deducted from the window budget."""
    buf = ChatSummaryMemoryBuffer(token_limit=1000, tokenizer_fn=words, llm=None)
    buf.system_prompt = " ".join(["sys"] * 100)
    buf.reserved_tokens = 200
    buf.summary = " ".join(["sum"] * 50)
    summary_cost = buf._token_count_for_message(buf._summary_message())
    assert buf._budget() == 1000 - (100 + 4) - 200 - summary_cost


def test_budget_counts_tool_arguments_and_thinking():
    """Confirmed: tool-call arguments and echoed thinking parts are counted."""
    from gwenflow.types import ThinkingContent

    buf = ChatMemoryBuffer(token_limit=1000, tokenizer_fn=words)
    bare = Message(role="assistant", content="ok")
    with_call = Message(
        role="assistant", content="ok", tool_calls=[{"id": "1", "function": {"arguments": "a b c d e"}}]
    )
    with_thinking = Message(
        role="assistant", content="ok", thinking_parts=[ThinkingContent(content="let me think hard")]
    )
    assert buf._token_count_for_message(with_call) > buf._token_count_for_message(bare)
    assert buf._token_count_for_message(with_thinking) > buf._token_count_for_message(bare)


def test_reserved_tokens_include_the_final_result_tool():
    """Confirmed: with a response_model and tools, the extra `final_result` schema is reserved."""
    from pydantic import BaseModel

    class Answer(BaseModel):
        text: str

    plain = scripted_agent()
    structured = scripted_agent(response_model=Answer)
    plain._prepare_history(task="t")
    structured._prepare_history(task="t")
    assert structured.history.reserved_tokens > plain.history.reserved_tokens


# ---------------------------------------------------------------------------
# 3. Accuracy of the local estimate: see the audit report (live measurement
#    against Mistral: -16 % on French prose, +8 % on code, +18 % on JSON).
# ---------------------------------------------------------------------------


def test_calibration_only_scales_the_whole_budget():
    """Confirmed: `token_scale` is a single global ratio, not per content type."""
    buf = ChatMemoryBuffer(token_limit=1000, tokenizer_fn=words)
    window = [Message(role="user", content="word " * 400)]
    estimated = buf._token_count_for_messages(window)
    buf.observe_prompt_tokens(window, 2 * estimated)  # the provider billed twice our estimate
    assert buf._effective_token_limit() == 500


# ---------------------------------------------------------------------------
# 4. The live turn
# ---------------------------------------------------------------------------


def test_live_turn_overflow_clears_results_then_drops_oldest_messages():
    """Confirmed: a long tool loop first loses old results, then whole messages, and is never summarized."""
    buf = ChatSummaryMemoryBuffer(token_limit=900, tokenizer_fn=words, llm=None)
    buf.add_messages(agent_history(n_reads=20, result_words=200, with_notes=True))
    buf.compact()
    window = buf.get()
    assert buf.summary == ""  # the live turn is never summarized
    notes = [m.content for m in window if m.role == "assistant" and m.content]
    assert "NOTE 0:" not in " ".join(notes)  # the oldest notes are gone, with no trace
    assert window[0].content.startswith("TASK:")  # but the task is kept


def test_intermediate_notes_of_a_long_turn_survive_in_some_form():
    """Older notes of the live turn reach the model, summarized if need be (fixed in step 2)."""
    llm = ScriptedLLM(summary="SUMMARY: notes 0 to 9 say the first files were relevant.")
    buf = ChatSummaryMemoryBuffer(token_limit=900, tokenizer_fn=words, llm=llm)
    buf.add_messages(agent_history(n_reads=20, result_words=200, with_notes=True))
    buf.compact()
    rendered = " ".join(str(m.content) for m in buf.get())
    assert "NOTE 0:" in rendered or "SUMMARY" in rendered


# ---------------------------------------------------------------------------
# 5. Tool call / result pairing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("token_limit", [300, 450, 600, 900, 1500])
def test_pruning_keeps_parallel_groups_whole(token_limit):
    """Confirmed: whatever the cut, a window never separates a call from its results."""
    messages = [Message(role="user", content="TASK: compare the markets.")]
    for i in range(6):
        messages += parallel_turn(f"g{i}_", 3, 60)
    buf = ChatMemoryBuffer(token_limit=token_limit, tokenizer_fn=words)
    buf.add_messages(messages)
    assert_tool_pairs(buf.get())


def test_clearing_and_summarizing_keep_pairs_whole():
    """Confirmed: stubs keep their tool_call_id, and the summary boundary is a group boundary."""
    llm = ScriptedLLM()
    buf = ChatSummaryMemoryBuffer(token_limit=700, tokenizer_fn=words, llm=llm)
    buf.add_messages(
        [Message(role="user", content="earlier question"), Message(role="assistant", content="earlier answer")]
    )
    buf.add_messages(agent_history(n_reads=12, result_words=200))
    buf.compact()
    window = buf.get()
    assert_tool_pairs([m for m in window if m.role != "system"])
    assert_tool_pairs(buf.messages)  # the stored history too: the cut was a group boundary


# ---------------------------------------------------------------------------
# 6. The initial instruction
# ---------------------------------------------------------------------------


def test_task_survives_as_long_as_it_is_the_last_user_message():
    """Confirmed: the anchor is the last user message, which is the task in a single-task run."""
    buf = ChatMemoryBuffer(token_limit=500, tokenizer_fn=words)
    buf.add_messages(agent_history(n_reads=20, result_words=200))
    assert buf.get()[0].content.startswith("TASK:")


def test_task_survives_a_validation_retry_message():
    """A schema-validation retry adds a user message: the task is still sent (fixed in step 2)."""
    messages = agent_history(n_reads=20, result_words=200)
    messages.append(Message(role="user", content="Validation error: field 'total' is required. Answer again."))
    buf = ChatMemoryBuffer(token_limit=500, tokenizer_fn=words)
    buf.add_messages(messages)
    assert any(str(m.content).startswith("TASK:") for m in buf.get())


# ---------------------------------------------------------------------------
# 7. Prefix stability (prompt cache)
# ---------------------------------------------------------------------------


def sent_prefix(buf: ChatMemoryBuffer) -> list[str]:
    """The window up to the first intact tool result.

    That is the part a prompt cache could reuse if it did not change.
    """
    prefix: list[str] = []
    for m in buf.get():
        if m.role == "tool" and str(m.content).startswith("data"):
            break
        prefix.append(f"{m.role}:{str(m.content)[:60]}")
    return prefix


@pytest.mark.xfail(strict=True, reason="the clearing frontier advances one result per request (step 2: hysteresis)")
def test_prefix_is_stable_between_two_compaction_batches():
    """Wanted: consecutive requests keep an identical prefix until the next batch."""
    buf = ChatMemoryBuffer(token_limit=1800, tokenizer_fn=words)
    buf.add_messages(agent_history(n_reads=12, result_words=200))
    prefixes = [sent_prefix(buf)]
    for i in range(3):
        buf.add_messages(tool_turn(f"extra{i}", "ReadFileTool", {"file_path": f"x{i}.csv"}, 200))
        prefixes.append(sent_prefix(buf))
    stable_steps = sum(1 for a, b in zip(prefixes, prefixes[1:], strict=False) if a == b)
    assert stable_steps >= 2


def test_prefix_currently_changes_on_every_request():
    """Confirmed defect, kept as documentation: each request clears one more result."""
    buf = ChatMemoryBuffer(token_limit=1800, tokenizer_fn=words)
    buf.add_messages(agent_history(n_reads=12, result_words=200))
    prefixes = [sent_prefix(buf)]
    for i in range(3):
        buf.add_messages(tool_turn(f"extra{i}", "ReadFileTool", {"file_path": f"x{i}.csv"}, 200))
        prefixes.append(sent_prefix(buf))
    assert all(a != b for a, b in zip(prefixes, prefixes[1:], strict=False))
    assert [len(p) for p in prefixes] == sorted(len(p) for p in prefixes)  # one more stub each time


# ---------------------------------------------------------------------------
# 8. Cost and failure of the summary
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=True, reason="compaction usage lives only in history.usage (step 3)")
def test_summary_tokens_are_part_of_the_run_usage():
    agent = scripted_agent(llm=ScriptedLLM(context_size=400, tool_turns=8))
    agent.run("TASK: first run.")
    agent.run("TASK: second run.")
    response = agent.run("TASK: third run.")
    # Agent copies its llm (one copy for the loop, one for the summarizer):
    # the counters live on the copies, the `prompts` list is shared.
    summary_calls = agent.history.llm.summary_calls
    assert summary_calls >= 1
    assert agent.total_usage.requests == len(agent.llm.prompts) + summary_calls
    assert response.usage.requests >= 1


def test_summary_usage_is_tracked_on_the_history_only():
    """Confirmed: the summarizer's spend is recorded, but beside the run usage."""
    agent = scripted_agent(llm=ScriptedLLM(context_size=400, tool_turns=8))
    for i in range(3):
        agent.run(f"TASK: run {i}.")
    summary_calls = agent.history.llm.summary_calls
    assert summary_calls >= 1
    assert agent.history.usage.requests == summary_calls
    assert agent.total_usage.requests == len(agent.llm.prompts)


def test_empty_summary_is_retried_a_bounded_number_of_times():
    """One retry, then the sliding window takes over (fixed in step 2)."""

    class EmptyLLM:
        calls = 0

        def invoke(self, input):
            self.calls += 1
            return ModelResponse(parts=[TextContent(content="")])

    llm = EmptyLLM()
    buf = ChatSummaryMemoryBuffer(token_limit=120, tokenizer_fn=words, llm=llm)
    for i in range(6):
        buf.add_message(Message(role="user", content=f"question {i} " + "blabla " * 15))
        buf.add_message(Message(role="assistant", content=f"answer {i} " + "detail " * 15))
    for _ in range(6):
        buf.compact()
    assert llm.calls <= 3


def test_failed_summary_degrades_to_dropping_without_losing_the_turn():
    """Confirmed: a broken summarizer never fails the run; the window just drops."""

    class BrokenLLM:
        def invoke(self, input):
            raise RuntimeError("boom")

    buf = ChatSummaryMemoryBuffer(token_limit=500, tokenizer_fn=words, llm=BrokenLLM())
    buf.add_messages([Message(role="user", content="earlier"), Message(role="assistant", content="x " * 100)])
    buf.add_messages(agent_history(n_reads=10, result_words=200))
    buf.compact()
    window = buf.get()
    assert buf.summary == ""
    assert any(str(m.content).startswith("TASK:") for m in window)
