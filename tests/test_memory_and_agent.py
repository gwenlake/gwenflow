"""Unit tests for ChatSummaryMemoryBuffer and the agent slash commands.

Covers:
- `get()` stays a pure read (no LLM call, no mutation of the stored history)
- `compact()` folds evicted messages into a rolling summary, iterating to a
  fixed point, incrementally, and retrying after an LLM failure
- ChatMemoryBuffer keeps its plain drop behaviour (compact/acompact are no-ops)
- Agent wiring: dedicated summarizer, compaction before every `get()`
- Slash commands (/help, /reset, /compact, /cost, /tools, /save, /load, ...)

The buffers are driven with a word-count tokenizer so the token budgets in the
tests stay readable; the production tokenizer is exercised in
`test_memory_buffer.py`.
"""

import asyncio
import json

import pytest

from gwenflow.agents.agent import Agent
from gwenflow.memory import ChatMemoryBuffer, ChatSummaryMemoryBuffer
from gwenflow.tools import Tool
from gwenflow.types import (
    AgentEventCompleted,
    AgentEventContent,
    AgentEventStarted,
    AgentResponse,
    Message,
    ModelResponse,
    RequestUsage,
    TextContent,
)


def tok(text: str) -> int:
    """Word-count tokenizer: cheap, offline, and easy to reason about."""
    return len(text.split())


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeLLM:
    """Summarizer stub: counts its calls and returns a new summary each time."""

    def __init__(self):
        self.calls = 0

    def invoke(self, input):
        self.calls += 1
        return ModelResponse(parts=[TextContent(content=f"RESUME v{self.calls} : questions numerotees du user.")])


class BrokenLLM:
    def __init__(self):
        self.calls = 0

    def invoke(self, input):
        self.calls += 1
        raise RuntimeError("boom")


class MiniLLM:
    """Fake LLM covering the surface Agent uses, summarizer included.

    Answers the compaction prompt with a summary and anything else with a
    canned agent reply, so a single instance can play both roles.
    """

    def __init__(self, context_size: int = 90):
        self.tools = []
        self.tool_choice = None
        self.response_format = None
        self.context_size = context_size
        self.main_inputs = []
        self.summary_calls = 0

    def get_context_size(self):
        return self.context_size

    def input_to_message_list(self, input):
        if isinstance(input, str):
            return [Message(role="user", content=input)]
        return [m if isinstance(m, Message) else Message(**m) for m in input]

    def get_thinking_parts(self, response):
        return None

    def _respond(self, input):
        sys_msg = input[0].get("content", "") if input and isinstance(input[0], dict) else ""
        if sys_msg.startswith("You maintain a running summary"):
            self.summary_calls += 1
            return ModelResponse(
                parts=[
                    TextContent(content=f"RESUME e2e n{self.summary_calls} : le user pose des questions numerotees.")
                ],
                usage=RequestUsage(input_tokens=100, output_tokens=10),
            )
        self.main_inputs.append(input)
        return ModelResponse(
            parts=[TextContent(content="reponse de l'agent " + "detail " * 10)],
            usage=RequestUsage(input_tokens=100, output_tokens=10),
        )

    def invoke(self, input):
        return self._respond(input)

    async def ainvoke(self, input):
        return self.invoke(input)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def summary_buffer(token_limit=120, llm=None):
    return ChatSummaryMemoryBuffer(token_limit=token_limit, tokenizer_fn=tok, llm=llm or FakeLLM())


def filled(buf, n=6):
    for i in range(n):
        buf.add_message(Message(role="user", content=f"question {i} " + "blabla " * 15))
        buf.add_message(Message(role="assistant", content=f"reponse {i} " + "detail " * 15))
    return buf


def make_agent(**kwargs) -> Agent:
    """Agent on a MiniLLM, with the word-count tokenizer wired into its memory."""
    kwargs.setdefault("llm", MiniLLM())
    agent = Agent(**kwargs)
    agent.history.tokenizer_fn = tok
    return agent


def run_a_few_turns(agent: Agent, n: int = 5) -> None:
    for i in range(n):
        agent.run(f"question {i} " + "blabla " * 12)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_llm_parameter_is_accepted():
    llm = FakeLLM()
    assert summary_buffer(llm=llm).llm is llm


def test_legacy_summarize_llm_parameter_is_rejected():
    with pytest.raises(TypeError):
        ChatSummaryMemoryBuffer(token_limit=120, summarize_llm=FakeLLM())


# ---------------------------------------------------------------------------
# get() is a pure read
# ---------------------------------------------------------------------------


def test_get_never_calls_the_llm():
    llm = FakeLLM()
    buf = filled(summary_buffer(llm=llm))
    buf.get()
    buf.get()
    assert llm.calls == 0


def test_get_is_idempotent():
    buf = filled(summary_buffer())
    assert [m.role for m in buf.get()] == [m.role for m in buf.get()]


def test_no_summary_before_compact():
    buf = filled(summary_buffer())
    buf.get()
    assert buf.summary == ""


# ---------------------------------------------------------------------------
# compact()
# ---------------------------------------------------------------------------


def test_compact_reaches_a_fixed_point_in_one_call():
    llm = FakeLLM()
    buf = filled(summary_buffer(llm=llm))
    buf.compact()
    calls_first = llm.calls
    buf.compact()
    assert calls_first >= 1
    assert llm.calls == calls_first


def test_summary_is_the_first_message_of_the_window():
    buf = filled(summary_buffer())
    buf.compact()
    assert "Summary of the earlier" in (buf.get()[0].content or "")


def test_compact_leaves_the_stored_history_intact():
    buf = filled(summary_buffer())
    buf.compact()
    assert all(len(m.content) > 50 for m in buf.messages)


def test_get_evicts_nothing_unsummarized_after_compact():
    buf = filled(summary_buffer())
    buf.compact()
    budget = buf._budget()
    start, anchor, _ = buf._prune(buf._clamp_history(list(buf.messages), budget), budget)
    evict_end = start if anchor is None else min(start, anchor)
    assert evict_end <= buf._summarized_upto


def test_llm_failure_keeps_the_pointer_and_does_not_loop():
    broken = BrokenLLM()
    buf = filled(summary_buffer(llm=broken))
    buf.compact()
    assert buf._summarized_upto == 0
    assert buf.summary == ""
    assert broken.calls == 1


def test_compaction_is_retried_on_the_next_call():
    buf = filled(summary_buffer(llm=BrokenLLM()))
    buf.compact()
    buf.llm = FakeLLM()
    buf.compact()
    assert buf.summary != ""
    assert buf._summarized_upto > 0


def test_messages_without_text_advance_without_an_llm_call():
    llm = FakeLLM()
    buf = ChatSummaryMemoryBuffer(token_limit=30, tokenizer_fn=tok, llm=llm)
    for _ in range(4):
        buf.add_message(Message(role="assistant", content=None))
    buf.add_message(Message(role="user", content="question finale " + "mot " * 20))
    buf.compact()
    assert llm.calls == 0


def test_acompact_folds_and_get_prefixes_the_summary():
    buf = filled(summary_buffer())

    async def go():
        await buf.acompact()
        return buf.get()

    assert "Summary" in (asyncio.run(go())[0].content or "")


def test_reset_clears_summary_and_pointer():
    buf = filled(summary_buffer())
    buf.compact()
    buf.reset()
    assert buf.summary == ""
    assert buf._summarized_upto == 0
    assert buf.messages == []


# ---------------------------------------------------------------------------
# The base buffer is unchanged
# ---------------------------------------------------------------------------


def test_base_buffer_compact_is_a_noop():
    buf = filled(ChatMemoryBuffer(token_limit=120, tokenizer_fn=tok))
    before = [(m.role, m.content) for m in buf.get()]
    buf.compact()
    asyncio.run(buf.acompact())
    after = [(m.role, m.content) for m in buf.get()]
    assert before == after
    assert any(role == "user" for role, _ in after)
    assert not any("Summary" in (content or "") for _, content in after)


def test_base_buffer_has_no_llm_or_summary_attribute():
    buf = ChatMemoryBuffer(token_limit=120, tokenizer_fn=tok)
    assert not hasattr(buf, "llm")
    assert not hasattr(buf, "summary")


def test_get_does_not_mutate_a_clamped_message():
    buf = ChatMemoryBuffer(token_limit=200, tokenizer_fn=tok)
    buf.add_message(Message(role="user", content="q"))
    buf.add_message(Message(role="assistant", content="mot " * 500))
    buf.add_message(Message(role="user", content="s"))
    before = len(buf.messages[1].content)
    sent = buf.get()
    assert len(buf.messages[1].content) == before
    assert len([m for m in sent if m.role == "assistant"][0].content) < before


def test_exhausted_budget_still_sends_the_user_message():
    buf = ChatMemoryBuffer(token_limit=100, tokenizer_fn=tok)
    buf.system_prompt = "x " * 50
    buf.reserved_tokens = 60
    buf.add_message(Message(role="user", content="derniere question importante"))
    out = buf.get()
    assert buf.messages[0].content == "derniere question importante"
    assert out[-1].content == "derniere question importante"


def test_window_never_starts_with_an_orphan_tool_message():
    buf = ChatMemoryBuffer(token_limit=60, tokenizer_fn=tok)
    buf.add_message(Message(role="user", content="q1 " * 10))
    buf.add_message(
        Message(
            role="assistant",
            content=None,
            tool_calls=[{"id": "1", "type": "function", "function": {"name": "t", "arguments": "{}"}}],
        )
    )
    buf.add_message(Message(role="tool", tool_call_id="1", content="resultat " * 15))
    buf.add_message(Message(role="assistant", content="reponse " * 5))
    buf.add_message(Message(role="user", content="q2 finale"))
    assert [m.role for m in buf.get()][0] != "tool"


# ---------------------------------------------------------------------------
# Agent wiring
# ---------------------------------------------------------------------------


def test_agent_defaults_to_a_summary_buffer():
    assert isinstance(make_agent().history, ChatSummaryMemoryBuffer)


def test_summarizer_is_a_dedicated_copy_without_tools():
    llm = MiniLLM()
    agent = make_agent(llm=llm)
    assert agent.history.llm is not llm
    assert agent.history.llm.tools == []
    assert agent.history.llm.response_format is None


def test_agent_tools_never_reach_the_summarizer():
    llm = MiniLLM()
    agent = make_agent(llm=llm)
    llm.tools = ["fake-tool"]
    assert agent.history.llm.tools == []


def test_run_triggers_compaction_and_sends_the_summary():
    agent = make_agent()
    run_a_few_turns(agent)
    assert agent.history.llm.summary_calls >= 1
    assert agent.history.summary != ""

    last_input = agent.llm.main_inputs[-1]
    assert any("Summary of the earlier" in (m.get("content") or "") for m in last_input)
    assert last_input[-1]["role"] == "user"


def test_arun_compacts_through_acompact():
    agent = make_agent()

    async def go():
        for i in range(5):
            await agent.arun(f"question {i} " + "blabla " * 12)

    asyncio.run(go())
    assert agent.history.llm.summary_calls >= 1
    assert agent.history.summary != ""


def test_failed_validation_stores_the_answer_once():
    """The invalid answer lands in the history exactly once.

    It used to be added twice: once on the normal path and once again on the
    validation-retry branch, before the retry message.
    """
    from pydantic import BaseModel

    class Answer(BaseModel):
        city: str

    class RetryLLM(MiniLLM):
        def _respond(self, input):
            self.main_inputs.append(input)
            content = "pas du json" if len(self.main_inputs) == 1 else '{"city": "Rennes"}'
            return ModelResponse(parts=[TextContent(content=content)], usage=RequestUsage())

    agent = make_agent(llm=RetryLLM(context_size=10_000), response_model=Answer)
    response = agent.run("quelle ville ?")

    assert response.parsed == Answer(city="Rennes")
    assert [m.role for m in agent.history.messages] == ["user", "assistant", "user", "assistant"]
    assert agent.history.messages[1].content == "pas du json"


# ---------------------------------------------------------------------------
# Slash commands
# ---------------------------------------------------------------------------


def test_help_lists_the_commands():
    content = make_agent().run("/help").content
    assert "/compact" in content
    assert "/reset" in content
    assert "/history" in content
    assert "/cost" in content
    assert all(c in content for c in ("/save", "/load", "/config", "/tools"))


def test_a_command_costs_no_llm_call_and_leaves_no_trace():
    agent = make_agent()
    run_a_few_turns(agent)
    stored, calls = len(agent.history.messages), len(agent.llm.main_inputs)
    agent.run("/help")
    assert len(agent.llm.main_inputs) == calls
    assert len(agent.history.messages) == stored


def test_history_command_reports_stats():
    agent = make_agent()
    run_a_few_turns(agent)
    content = agent.run("/history").content
    assert "message(s) stored" in content
    assert "limit" in content


def test_compact_command_is_case_and_space_insensitive():
    agent = make_agent()
    run_a_few_turns(agent)
    before = agent.history._summarized_upto
    content = agent.run("  /COMPACT  ").content
    assert ("Compacted" in content and agent.history._summarized_upto >= before) or "Nothing to compact" in content


def test_reset_clears_history_and_summary():
    agent = make_agent()
    run_a_few_turns(agent)
    stored = len(agent.history.messages)
    content = agent.run("/reset").content
    assert f"{stored} message(s) removed" in content
    assert agent.history.messages == []
    assert agent.history.summary == ""


def test_clear_is_an_alias_of_reset():
    agent = make_agent()
    agent.run("nouvelle question " + "mot " * 10)
    assert "removed" in agent.run("/clear").content
    assert agent.history.messages == []


def test_a_command_followed_by_text_is_not_intercepted():
    agent = make_agent()
    agent.run("test " + "mot " * 5)
    messages, calls = len(agent.history.messages), len(agent.llm.main_inputs)
    agent.run("/reset the counter in my code please")
    assert len(agent.llm.main_inputs) == calls + 1
    assert len(agent.history.messages) > messages


def test_an_unknown_command_goes_to_the_llm():
    agent = make_agent()
    agent.run("/foobar")
    assert len(agent.llm.main_inputs) == 1


def test_compact_on_a_plain_buffer_reports_it_is_unsupported():
    agent = Agent(name="cmd-base", llm=MiniLLM(), history=ChatMemoryBuffer(token_limit=90, tokenizer_fn=tok))
    assert "does not support compaction" in agent.run("/compact").content


def test_arun_compact_command():
    agent = make_agent()

    async def go():
        for i in range(5):
            await agent.arun(f"question {i} " + "blabla " * 12)
        return await agent.arun("/compact")

    content = asyncio.run(go()).content
    assert "Compacted" in content or "Nothing to compact" in content


def test_run_stream_emits_events_and_a_final_response():
    items = list(make_agent().run_stream("/help"))
    kinds = [type(x).__name__ for x in items]
    assert kinds[:3] == [AgentEventStarted.__name__, AgentEventContent.__name__, AgentEventCompleted.__name__]
    assert isinstance(items[-1], AgentResponse)
    assert "/compact" in items[-1].content


# ---------------------------------------------------------------------------
# /cost
# ---------------------------------------------------------------------------


@pytest.fixture
def costly_agent():
    agent = make_agent(name="cost")
    run_a_few_turns(agent)
    return agent


def test_total_usage_accumulates_over_runs(costly_agent):
    calls = len(costly_agent.llm.main_inputs)
    assert costly_agent.total_usage.requests == calls
    assert costly_agent.total_usage.input_tokens == 100 * calls
    assert costly_agent.total_usage.output_tokens == 10 * calls


def test_compaction_usage_is_counted_separately(costly_agent):
    summary_calls = costly_agent.history.llm.summary_calls
    assert summary_calls >= 1
    assert costly_agent.history.usage.requests == summary_calls
    assert costly_agent.history.usage.input_tokens == 100 * summary_calls


def test_cost_without_pricing(costly_agent):
    main_calls = len(costly_agent.llm.main_inputs)
    content = costly_agent.run("/cost").content
    assert "Session usage" in content
    assert "Compaction" in content
    assert "unavailable" in content
    assert f"{100 * main_calls:,}" in content
    # the command itself is free and not counted as a request
    assert len(costly_agent.llm.main_inputs) == main_calls
    assert costly_agent.total_usage.requests == main_calls


def test_cost_with_pricing(costly_agent):
    calls = len(costly_agent.llm.main_inputs) + costly_agent.history.llm.summary_calls
    costly_agent.pricing = {"input": 1.0, "output": 10.0}
    expected = (calls * 100 * 1.0 + calls * 10 * 10.0) / 1_000_000
    content = costly_agent.run("/cost").content
    assert f"${expected:.4f}" in content
    assert "unavailable" not in content


def test_reset_does_not_zero_the_cost(costly_agent):
    main_calls = len(costly_agent.llm.main_inputs)
    summary_calls = costly_agent.history.llm.summary_calls
    costly_agent.run("/reset")
    costly_agent.run("/cost")
    assert costly_agent.total_usage.requests == main_calls
    assert costly_agent.history.usage.requests == summary_calls


# ---------------------------------------------------------------------------
# /tools /summary /system /config
# ---------------------------------------------------------------------------


def meteo(ville: str) -> str:
    """Donne la meteo d'une ville."""
    return "beau temps"


@pytest.fixture
def tooled_agent():
    return make_agent(name="lot2", instructions=["Reponds en francais"], tools=[Tool(meteo)])


def test_tools_command_lists_the_tools(tooled_agent):
    content = tooled_agent.run("/tools").content
    assert "meteo" in content
    assert "1 tool(s)" in content


def test_tools_command_without_tools():
    assert "No tools registered" in make_agent(name="vide").run("/tools").content


def test_tools_command_followed_by_text_is_not_intercepted(tooled_agent):
    calls = len(tooled_agent.llm.main_inputs)
    tooled_agent.run("/tools et dis moi la meteo")
    assert len(tooled_agent.llm.main_inputs) == calls + 1


def test_summary_command_is_empty_before_compaction(tooled_agent):
    assert "empty" in tooled_agent.run("/summary").content


def test_summary_command_shows_the_summary(tooled_agent):
    run_a_few_turns(tooled_agent)
    assert "RESUME" in tooled_agent.run("/summary").content


def test_summary_command_on_a_plain_buffer():
    agent = Agent(name="base", llm=MiniLLM(), history=ChatMemoryBuffer(token_limit=90, tokenizer_fn=tok))
    assert "does not keep a summary" in agent.run("/summary").content


def test_system_command_shows_the_instructions(tooled_agent):
    assert "Reponds en francais" in tooled_agent.run("/system").content


def test_system_command_with_a_fixed_system_prompt():
    content = make_agent(name="fixe", system_prompt="Tu es un pirate.").run("/system").content
    assert "Tu es un pirate." in content
    assert "(fixed)" in content


def test_config_command(tooled_agent):
    content = tooled_agent.run("/config").content
    assert "lot2" in content
    assert "ChatSummaryMemoryBuffer" in content
    assert "max_turns" in content
    assert "tools:           1" in content


# ---------------------------------------------------------------------------
# /save and /load
# ---------------------------------------------------------------------------


def test_save_without_an_argument_shows_the_usage():
    assert "Usage: /save" in make_agent().run("/save").content


def test_save_load_round_trip(tooled_agent, tmp_path):
    run_a_few_turns(tooled_agent)
    path = tmp_path / "session.json"
    count = len(tooled_agent.history.messages)
    saved_summary = tooled_agent.history.summary
    saved_upto = tooled_agent.history._summarized_upto

    content = tooled_agent.run(f"/save {path}").content
    assert path.exists()
    assert str(path) in content
    assert f"{count} message(s)" in content

    data = json.loads(path.read_text())
    assert data["format"] == "gwenflow.session"
    assert data["version"] == 1
    assert len(data["messages"]) == count
    assert data["summary"] == saved_summary

    tooled_agent.run("/reset")
    assert tooled_agent.history.messages == []
    content = tooled_agent.run(f"/load {path}").content
    assert f"{count} message(s)" in content
    assert "summary restored" in content
    assert len(tooled_agent.history.messages) == count
    assert tooled_agent.history.summary == saved_summary
    assert tooled_agent.history._summarized_upto == saved_upto

    # and the agent keeps working on the restored history
    tooled_agent.run("nouvelle question apres restauration " + "mot " * 8)
    assert len(tooled_agent.history.messages) == count + 2


def test_load_missing_file(tmp_path):
    agent = make_agent()
    assert "Could not load session" in agent.run(f"/load {tmp_path / 'absent.json'}").content


def test_load_a_json_that_is_not_a_session(tmp_path):
    path = tmp_path / "pas_une_session.json"
    path.write_text('{"hello": 1}')
    assert "not a gwenflow session" in make_agent().run(f"/load {path}").content


def test_load_refuses_a_future_format_version(tmp_path):
    path = tmp_path / "future.json"
    path.write_text('{"format": "gwenflow.session", "version": 99, "messages": []}')
    assert "newer than supported" in make_agent().run(f"/load {path}").content


def test_save_on_a_plain_buffer_writes_no_summary(tmp_path):
    agent = Agent(name="base-save", llm=MiniLLM(), history=ChatMemoryBuffer(token_limit=200, tokenizer_fn=tok))
    path = tmp_path / "base_session.json"
    agent.run("bonjour " + "mot " * 5)
    agent.run(f"/save {path}")
    assert "summary" not in json.loads(path.read_text())


def test_load_on_a_plain_buffer_ignores_the_summary(tooled_agent, tmp_path):
    run_a_few_turns(tooled_agent)
    path = tmp_path / "session.json"
    tooled_agent.run(f"/save {path}")

    agent = Agent(name="base-load", llm=MiniLLM(), history=ChatMemoryBuffer(token_limit=200, tokenizer_fn=tok))
    assert "summary in file ignored" in agent.run(f"/load {path}").content


def test_round_trip_preserves_tool_calls_and_multipart_content(tmp_path):
    agent = make_agent(name="rt")
    agent.history.add_message(Message(role="user", content="quelle meteo ?"))
    agent.history.add_message(
        Message(
            role="assistant",
            content=None,
            tool_calls=[
                {"id": "tc1", "type": "function", "function": {"name": "meteo", "arguments": '{"ville": "Rennes"}'}}
            ],
        )
    )
    agent.history.add_message(Message(role="tool", tool_call_id="tc1", content="beau temps"))
    agent.history.add_message(Message(role="assistant", content=[TextContent(content="Il fait beau a Rennes.")]))
    before = [m.to_dict() for m in agent.history.messages]

    path = tmp_path / "rt_session.json"
    agent.run(f"/save {path}")
    agent.run("/reset")
    agent.run(f"/load {path}")

    assert [m.to_dict() for m in agent.history.messages] == before
    assert agent.history.messages[1].tool_calls[0]["id"] == "tc1"
    assert isinstance(agent.history.messages[3].content[0], TextContent)
