"""An answer cut short by the output token limit or the content filter.

The LLM clients report it as `finish_reason="length"` / `"content_filter"`
(any other reason is unchanged), and the agent ends the run with the same
reason instead of "stop". A tool call cut by the limit is not run: the model
is told and can make it again.
"""

import asyncio
import enum
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from gwenflow.agents.agent import CUT_SHORT_TOOL_CALL, Agent
from gwenflow.llms.base import ChatBase, normalize_finish_reason
from gwenflow.tools import BaseTool
from gwenflow.types import AgentResponse, ModelResponse, RequestUsage, TextContent, ToolCall

# ---------------------------------------------------------------------------
# normalize_finish_reason
# ---------------------------------------------------------------------------


class _GoogleFinishReason(enum.Enum):
    STOP = "STOP"
    MAX_TOKENS = "MAX_TOKENS"
    SAFETY = "SAFETY"


@pytest.mark.parametrize(
    "reason, expected",
    [
        ("length", "length"),
        ("model_length", "length"),
        ("max_tokens", "length"),
        ("max_output_tokens", "length"),
        ("content_filter", "content_filter"),
        ("refusal", "content_filter"),
        (_GoogleFinishReason.MAX_TOKENS, "length"),
        (_GoogleFinishReason.SAFETY, "content_filter"),
        ("FinishReason.MAX_TOKENS", "length"),
        (_GoogleFinishReason.STOP, None),
        ("stop", None),
        ("end_turn", None),
        ("tool_calls", None),
        ("tool_use", None),
        (None, None),
        ("something_new", None),
    ],
)
def test_normalize_finish_reason(reason, expected):
    assert normalize_finish_reason(reason) == expected


# ---------------------------------------------------------------------------
# ChatOpenAI (and every OpenAI-compatible client)
# ---------------------------------------------------------------------------


def _openai_completion(text="partial", finish_reason="stop", tool_calls=None):
    message = SimpleNamespace(role="assistant", content=text, tool_calls=tool_calls, reasoning_content=None)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    return SimpleNamespace(choices=[choice], usage=usage)


def _openai(monkeypatch, create_return):
    from gwenflow.llms.openai import ChatOpenAI

    client = MagicMock()
    client.chat.completions.create.return_value = create_return
    monkeypatch.setattr(ChatOpenAI, "get_client", lambda self: client)
    return ChatOpenAI(api_key="fake-key")


@pytest.mark.parametrize(
    "provider, expected", [("length", "length"), ("content_filter", "content_filter"), ("stop", "stop")]
)
def test_openai_invoke_reports_cut_short(monkeypatch, provider, expected):
    result = _openai(monkeypatch, _openai_completion(finish_reason=provider)).invoke("hi")
    assert result.finish_reason == expected
    assert result.content == "partial"


def test_openai_invoke_keeps_stop_for_tool_calls(monkeypatch):
    """Unchanged: the OpenAI client has always reported "stop" for a tool-call turn."""
    call = SimpleNamespace(id="c1", function=SimpleNamespace(name="add", arguments="{}"))
    result = _openai(monkeypatch, _openai_completion(text=None, finish_reason="tool_calls", tool_calls=[call])).invoke(
        "hi"
    )
    assert result.finish_reason == "stop"


def _openai_chunk(content=None, tool_calls=None, finish_reason=None):
    delta = SimpleNamespace(content=content, tool_calls=tool_calls, audio=None, reasoning_content=None)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=finish_reason)], usage=None)


def _openai_usage_chunk():
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    return SimpleNamespace(choices=[], usage=usage)


def test_openai_stream_reports_length_last(monkeypatch):
    chunks = [_openai_chunk("Once upon"), _openai_chunk(" a time", finish_reason="length"), _openai_usage_chunk()]
    responses = list(_openai(monkeypatch, iter(chunks)).stream("hi"))
    assert "".join(r.content or "" for r in responses) == "Once upon a time"
    assert responses[-1].finish_reason == "length"
    assert [r.finish_reason for r in responses[:-1]] == [None] * (len(responses) - 1)


def test_openai_stream_normal_end_adds_nothing(monkeypatch):
    chunks = [_openai_chunk("Hello"), _openai_chunk(finish_reason="stop"), _openai_usage_chunk()]
    responses = list(_openai(monkeypatch, iter(chunks)).stream("hi"))
    assert all(r.finish_reason is None for r in responses)
    assert len(responses) == 2  # the text and the usage, as before


def test_openai_stream_hands_over_tool_calls_cut_short(monkeypatch):
    delta = SimpleNamespace(id="c1", index=0, type="function", function=SimpleNamespace(name="add", arguments='{"a"'))
    chunks = [_openai_chunk(tool_calls=[delta]), _openai_chunk(finish_reason="length"), _openai_usage_chunk()]
    responses = list(_openai(monkeypatch, iter(chunks)).stream("hi"))
    assert responses[-1].finish_reason == "length"
    assert [t.arguments for t in responses[-1].tool_calls] == ['{"a"']


# ---------------------------------------------------------------------------
# ChatMistral
# ---------------------------------------------------------------------------


def test_mistral_parse_response_reports_length():
    from gwenflow.llms.mistral import ChatMistral

    msg = SimpleNamespace(role="assistant", content="partial", tool_calls=None)
    choice = SimpleNamespace(index=0, message=msg, finish_reason="length")
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    response = SimpleNamespace(id="r1", choices=[choice], usage=usage, object=None, created=None, model="m")
    assert ChatMistral(api_key="test")._parse_response(response).finish_reason == "length"


def test_mistral_stream_chunk_reports_length_and_hands_over_tool_calls():
    from gwenflow.llms.mistral import ChatMistral

    delta = SimpleNamespace(content=None, tool_calls=None)
    chunk = SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason="length")], usage=None)
    pending = [ToolCall(id="c1", name="add", arguments='{"a"')]
    response = ChatMistral(api_key="test")._process_stream_chunk(SimpleNamespace(data=chunk), pending)
    assert response.finish_reason == "length"
    assert [t.id for t in response.tool_calls] == ["c1"]


# ---------------------------------------------------------------------------
# ChatAnthropic
# ---------------------------------------------------------------------------


def _anthropic(monkeypatch):
    from gwenflow.llms.anthropic import ChatAnthropic

    client = MagicMock()
    monkeypatch.setattr(ChatAnthropic, "get_client", lambda self: client)
    return ChatAnthropic(api_key="fake-key"), client


@pytest.mark.parametrize(
    "stop_reason, expected",
    [("max_tokens", "length"), ("refusal", "content_filter"), ("end_turn", "stop"), ("tool_use", "tool_calls")],
)
def test_anthropic_invoke_reports_cut_short(monkeypatch, stop_reason, expected):
    llm, client = _anthropic(monkeypatch)
    client.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="partial")],
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )
    assert llm.invoke("hi").finish_reason == expected


def _anthropic_stream(client, events):
    client.messages.stream.return_value.__enter__.return_value = iter(events)


def test_anthropic_stream_reports_length(monkeypatch):
    llm, client = _anthropic(monkeypatch)
    _anthropic_stream(
        client,
        [
            SimpleNamespace(type="content_block_delta", index=0, delta=SimpleNamespace(type="text_delta", text="part")),
            SimpleNamespace(
                type="message_delta",
                delta=SimpleNamespace(stop_reason="max_tokens"),
                usage=SimpleNamespace(output_tokens=5),
            ),
        ],
    )
    responses = list(llm.stream("hi"))
    assert responses[-1].finish_reason == "length"
    assert responses[-1].usage.output_tokens == 5


def test_anthropic_stream_hands_over_tool_calls_cut_short(monkeypatch):
    llm, client = _anthropic(monkeypatch)
    block = SimpleNamespace(type="tool_use", id="tu_1", name="add")
    _anthropic_stream(
        client,
        [
            SimpleNamespace(type="content_block_start", index=0, content_block=block),
            SimpleNamespace(
                type="content_block_delta",
                index=0,
                delta=SimpleNamespace(type="input_json_delta", partial_json='{"a"'),
            ),
            SimpleNamespace(
                type="message_delta",
                delta=SimpleNamespace(stop_reason="max_tokens"),
                usage=SimpleNamespace(output_tokens=0),
            ),
        ],
    )
    responses = list(llm.stream("hi"))
    assert responses[-1].finish_reason == "length"
    assert [t.arguments for t in responses[-1].tool_calls] == ['{"a"']


def test_anthropic_stream_normal_end_unchanged(monkeypatch):
    llm, client = _anthropic(monkeypatch)
    _anthropic_stream(
        client,
        [
            SimpleNamespace(type="content_block_delta", index=0, delta=SimpleNamespace(type="text_delta", text="hi")),
            SimpleNamespace(
                type="message_delta",
                delta=SimpleNamespace(stop_reason="end_turn"),
                usage=SimpleNamespace(output_tokens=0),
            ),
        ],
    )
    responses = list(llm.stream("hi"))
    assert len(responses) == 1
    assert responses[0].finish_reason is None


# ---------------------------------------------------------------------------
# ChatGoogle
# ---------------------------------------------------------------------------


def _google_candidate(text, finish_reason):
    part = SimpleNamespace(thought=False, text=text, function_call=None, inline_data=None)
    return SimpleNamespace(content=SimpleNamespace(parts=[part]), finish_reason=finish_reason)


def test_google_parse_response_reports_length():
    from gwenflow.llms.google import ChatGoogle

    response = SimpleNamespace(
        candidates=[_google_candidate("partial", _GoogleFinishReason.MAX_TOKENS)], usage_metadata=None
    )
    result = ChatGoogle(api_key="test")._parse_response(response)
    assert result.finish_reason == "length"
    assert result.content == "partial"


def test_google_stream_chunk_reports_safety():
    from gwenflow.llms.google import ChatGoogle

    chunk = SimpleNamespace(candidates=[_google_candidate("", _GoogleFinishReason.SAFETY)], usage_metadata=None)
    assert ChatGoogle(api_key="test")._stream_chunk_to_response(chunk).finish_reason == "content_filter"


def test_google_stream_chunk_normal_stop_unchanged():
    from gwenflow.llms.google import ChatGoogle

    chunk = SimpleNamespace(candidates=[_google_candidate("hi", _GoogleFinishReason.STOP)], usage_metadata=None)
    assert ChatGoogle(api_key="test")._stream_chunk_to_response(chunk).finish_reason is None


# ---------------------------------------------------------------------------
# ResponseOpenAI
# ---------------------------------------------------------------------------


def _responses_api(status="completed", reason=None, text="partial"):
    message = SimpleNamespace(type="message", content=[SimpleNamespace(type="output_text", text=text)])
    return SimpleNamespace(
        status=status,
        incomplete_details=SimpleNamespace(reason=reason) if reason else None,
        output=[message],
        usage=None,
    )


def test_responses_incomplete_on_max_output_tokens_returns_the_partial_answer():
    from gwenflow.llms.openai_response import ResponseOpenAI

    result = ResponseOpenAI(api_key="test")._parse_response(_responses_api("incomplete", "max_output_tokens"))
    assert result.finish_reason == "length"
    assert result.content == "partial"


def test_responses_incomplete_for_another_reason_still_raises():
    from gwenflow.llms.openai_response import ResponseOpenAI

    with pytest.raises(RuntimeError, match="incomplete"):
        ResponseOpenAI(api_key="test")._parse_response(_responses_api("incomplete", "something_else"))


def test_responses_stream_incomplete_on_max_output_tokens(monkeypatch):
    from gwenflow.llms.openai_response import ResponseOpenAI

    client = MagicMock()
    client.responses.stream.return_value.__enter__.return_value = iter(
        [
            SimpleNamespace(type="response.output_text.delta", delta="part"),
            SimpleNamespace(type="response.incomplete", response=_responses_api("incomplete", "max_output_tokens")),
        ]
    )
    monkeypatch.setattr(ResponseOpenAI, "get_client", lambda self: client)
    responses = list(ResponseOpenAI(api_key="test").stream("hi"))
    assert responses[0].content == "part"
    assert responses[-1].finish_reason == "length"


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class AddTool(BaseTool):
    name: str = "add"
    description: str = "Adds two numbers"
    calls: int = 0

    def _run(self, a: int, b: int) -> int:
        self.calls += 1
        return a + b


def _turn(content=None, tool_calls=None, finish_reason="stop"):
    parts = [TextContent(content=content)] if content else []
    parts += tool_calls or []
    return ModelResponse(parts=parts, finish_reason=finish_reason, usage=RequestUsage(input_tokens=10, output_tokens=5))


def _scripted_llm(turns):
    """An LLM that plays `turns` in order, on every path (invoke, ainvoke, stream, astream)."""
    script = iter(turns)

    def invoke(input=None, **kwargs):
        return next(script)

    async def ainvoke(input=None, **kwargs):
        return next(script)

    def stream(input=None, **kwargs):
        turn = next(script)
        if turn.content:
            yield ModelResponse(parts=[TextContent(content=turn.content)])
        yield ModelResponse(parts=list(turn.tool_calls), usage=turn.usage, finish_reason=_streamed(turn))

    async def astream(input=None, **kwargs):
        for chunk in stream(input):
            yield chunk

    llm = MagicMock()
    llm.get_context_size.return_value = 96000
    llm.response_format = None
    llm.tools = []
    llm.tool_choice = None
    llm.invoke.side_effect = invoke
    llm.ainvoke.side_effect = ainvoke
    llm.stream.side_effect = stream
    llm.astream.side_effect = astream
    llm.input_to_message_list.side_effect = ChatBase.input_to_message_list.__get__(llm)
    llm.get_thinking_parts.return_value = None
    return llm


def _streamed(turn):
    # The clients only stream an abnormal end.
    return turn.finish_reason if turn.finish_reason in ("length", "content_filter") else None


def _run(agent, path, task="go"):
    """Run `agent` on `path` and return the AgentResponse."""
    if path == "run":
        return agent.run(task)
    if path == "arun":
        return asyncio.run(agent.arun(task))
    if path == "run_stream":
        return [e for e in agent.run_stream(task) if isinstance(e, AgentResponse)][-1]

    async def consume():
        return [e async for e in agent.arun_stream(task) if isinstance(e, AgentResponse)][-1]

    return asyncio.run(consume())


PATHS = ["run", "arun", "run_stream", "arun_stream"]


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("reason", ["length", "content_filter"])
def test_run_ends_with_the_reason_the_answer_was_cut_short(path, reason):
    agent = Agent(llm=_scripted_llm([_turn("Once upon a", finish_reason=reason)]))
    response = _run(agent, path)
    assert response.finish_reason == reason
    assert response.content == "Once upon a"


@pytest.mark.parametrize("path", PATHS)
def test_run_complete_answer_still_stops(path):
    agent = Agent(llm=_scripted_llm([_turn("All done.")]))
    assert _run(agent, path).finish_reason == "stop"


@pytest.mark.parametrize("path", PATHS)
def test_a_cut_turn_before_the_final_answer_does_not_leak(path):
    """Only the final answer decides: a turn cut earlier in the run is not reported."""
    tool = AddTool()
    turns = [
        _turn(tool_calls=[ToolCall(id="c1", name="add", arguments='{"a": 1, "b": 2}')], finish_reason="length"),
        _turn("3"),
    ]
    response = _run(Agent(llm=_scripted_llm(turns), tools=[tool]), path)
    assert response.finish_reason == "stop"
    assert tool.calls == 1  # complete arguments: the call ran


@pytest.mark.parametrize("path", PATHS)
def test_tool_call_cut_by_the_limit_is_not_run(path):
    tool = AddTool()
    cut = ToolCall(id="c1", name="add", arguments='{"a": 1, "b"')
    whole = ToolCall(id="c2", name="add", arguments='{"a": 2, "b": 2}')
    agent = Agent(llm=_scripted_llm([_turn(tool_calls=[whole, cut], finish_reason="length"), _turn("4")]), tools=[tool])
    response = _run(agent, path)

    assert response.finish_reason == "stop"
    assert response.content == "4"
    assert tool.calls == 1
    results = {m.tool_call_id: m.content for m in agent.history.messages if m.role == "tool"}
    assert results["c1"] == CUT_SHORT_TOOL_CALL.format(name="add")
    assert results["c2"] == "4"
    completed = [e for e in response.events if e.event_type == "tool_completed"]
    assert {e.tool_call_id for e in completed} == {"c1", "c2"}


def test_invalid_arguments_without_truncation_keep_the_old_error():
    """Unchanged: malformed arguments in a turn that ended normally still reach the tool error path."""
    tool = AddTool()
    bad = ToolCall(id="c1", name="add", arguments='{"a": 1, "b"')
    agent = Agent(llm=_scripted_llm([_turn(tool_calls=[bad], finish_reason="tool_calls"), _turn("ok")]), tools=[tool])
    agent.run("go")
    result = next(m.content for m in agent.history.messages if m.role == "tool")
    assert result.startswith("Error executing tool 'add'")


class _Total(BaseModel):
    total: int


def test_structured_answer_cut_short_is_still_retried():
    """Unchanged: a truncated JSON answer gets the validation retries; the final turn decides the reason."""
    llm = _scripted_llm([_turn('{"tot', finish_reason="length"), _turn('{"total": 7}')])
    response = Agent(llm=llm, response_model=_Total).run("go")
    assert response.parsed == _Total(total=7)
    assert response.finish_reason == "stop"


def test_structured_answer_still_cut_after_retries_reports_length():
    llm = _scripted_llm([_turn('{"tot', finish_reason="length") for _ in range(4)])
    response = Agent(llm=llm, response_model=_Total).run("go")
    assert response.parsed is None
    assert response.finish_reason == "length"


def test_delegated_answer_cut_short_is_flagged_to_the_orchestrator():
    cut = AgentResponse(agent_id="w", content="half a report", finish_reason="length")
    result = Agent._delegation_result("Delegation", "writer", cut)
    assert result.startswith("half a report")
    assert "cut off by the output token limit" in result

    whole = AgentResponse(agent_id="w", content="a report", finish_reason="stop")
    assert Agent._delegation_result("Delegation", "writer", whole) == "a report"


def _cut_turn(call_id, also=None):
    calls = [ToolCall(id=call_id, name="add", arguments='{"a": 1, "b"')] + (also or [])
    return _turn(tool_calls=calls, finish_reason="length")


@pytest.mark.parametrize("path", PATHS)
def test_run_gives_up_after_tool_calls_cut_twice_in_a_row(path):
    tool = AddTool()
    whole = ToolCall(id="w2", name="add", arguments='{"a": 2, "b": 2}')
    llm = _scripted_llm([_cut_turn("c1"), _cut_turn("c2", also=[whole]), _turn("never reached")])
    agent = Agent(llm=llm, tools=[tool])
    response = _run(agent, path)

    assert response.finish_reason == "length"
    assert response.content is None
    assert response.usage.requests == 2
    # The complete call of the last turn still ran: no call is left without a result.
    assert tool.calls == 1
    answered = {m.tool_call_id for m in agent.history.messages if m.role == "tool"}
    assert answered == {"c1", "c2", "w2"}
    assert agent._pending_tool_calls() == []


@pytest.mark.parametrize("path", PATHS)
def test_a_complete_turn_resets_the_count_of_cut_turns(path):
    tool = AddTool()
    whole = ToolCall(id="w1", name="add", arguments='{"a": 2, "b": 2}')
    llm = _scripted_llm([_cut_turn("c1"), _turn(tool_calls=[whole]), _cut_turn("c2"), _turn("done")])
    response = _run(Agent(llm=llm, tools=[tool]), path)
    assert response.finish_reason == "stop"
    assert response.content == "done"


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("arguments", [{}, {"a": 1}, "", None])
def test_last_call_that_nothing_proves_whole_is_not_run(path, arguments):
    """The limit fell while the last call was written: parsed by the provider, or with no arguments yet."""
    tool = AddTool()
    whole = ToolCall(id="c1", name="add", arguments={"a": 2, "b": 2})
    last = ToolCall(id="c2", name="add", arguments=arguments)
    agent = Agent(
        llm=_scripted_llm([_turn(tool_calls=[whole, last], finish_reason="length"), _turn("4")]), tools=[tool]
    )
    response = _run(agent, path)

    assert response.content == "4"
    assert tool.calls == 1  # the call before it was whole: the model had moved on
    results = {m.tool_call_id: m.content for m in agent.history.messages if m.role == "tool"}
    assert results == {"c1": "4", "c2": CUT_SHORT_TOOL_CALL.format(name="add")}


@pytest.mark.parametrize("path", PATHS)
def test_cut_call_keeps_valid_arguments_in_the_history(path):
    """Half-written JSON is not replayed to the provider; the events still show what the model wrote."""
    tool = AddTool()
    whole = ToolCall(id="c1", name="add", arguments='{"a": 2, "b": 2}')
    cut = ToolCall(id="c2", name="add", arguments='{"a": 1, "b"')
    agent = Agent(llm=_scripted_llm([_turn(tool_calls=[whole, cut], finish_reason="length"), _turn("4")]), tools=[tool])
    response = _run(agent, path)

    turn = next(m for m in agent.history.messages if m.role == "assistant" and m.tool_calls)
    assert [c["function"]["arguments"] for c in turn.tool_calls] == ['{"a": 2, "b": 2}', "{}"]
    started = {e.tool_call_id: e.tool_args for e in response.events if e.event_type == "tool_started"}
    assert started["c2"] == {"raw": '{"a": 1, "b"'}


@pytest.mark.parametrize("path", PATHS)
def test_valid_final_result_beside_a_cut_call_is_a_whole_answer(path):
    answer = ToolCall(id="c1", name="final_result", arguments='{"total": 3}')
    cut = ToolCall(id="c2", name="add", arguments='{"a": 1, "b"')
    llm = _scripted_llm([_turn(tool_calls=[answer, cut], finish_reason="length")])
    response = _run(Agent(llm=llm, tools=[AddTool()], response_model=_Total), path)
    assert response.parsed == _Total(total=3)
    assert response.finish_reason == "stop"


@pytest.mark.parametrize("path", PATHS)
def test_a_turn_without_tool_call_resets_the_count_of_cut_turns(path):
    """Two cut turns apart from each other, with a refused structured answer in between, are not in a row."""
    turns = [_cut_turn("c1"), _turn('{"tot'), _cut_turn("c2"), _turn('{"total": 7}')]
    response = _run(Agent(llm=_scripted_llm(turns), tools=[AddTool()], response_model=_Total), path)
    assert response.finish_reason == "stop"
    assert response.parsed == _Total(total=7)


@pytest.mark.parametrize("path", PATHS)
def test_results_of_calls_sharing_an_id_keep_their_own_arguments(path):
    """A provider that sends no id (Gemini) gives the calls of a turn the same one."""
    calls = [
        ToolCall(id="add", name="add", arguments='{"a": 1, "b": 2}'),
        ToolCall(id="add", name="add", arguments='{"a": 5, "b": 5}'),
    ]
    llm = _scripted_llm([_turn(tool_calls=calls, finish_reason="tool_calls"), _turn("done")])
    response = _run(Agent(llm=llm, tools=[AddTool()]), path)
    completed = [(e.tool_args, e.content) for e in response.events if e.event_type == "tool_completed"]
    assert completed == [({"a": 1, "b": 2}, "3"), ({"a": 5, "b": 5}, "10")]


def test_started_and_completed_events_show_the_same_arguments():
    odd = ToolCall(id="c1", name="add", arguments="[1, 2]")
    llm = _scripted_llm([_turn(tool_calls=[odd], finish_reason="tool_calls"), _turn("done")])
    response = Agent(llm=llm, tools=[AddTool()]).run("go")
    started = next(e for e in response.events if e.event_type == "tool_started")
    completed = next(e for e in response.events if e.event_type == "tool_completed")
    assert started.tool_args == completed.tool_args == {"raw": "[1, 2]"}


def test_openai_stream_cut_short_reason_comes_with_the_usage(monkeypatch):
    """The last chunk carries both: a tracer that reads the last chunk keeps the token counts."""
    delta = SimpleNamespace(id="c1", index=0, type="function", function=SimpleNamespace(name="add", arguments='{"a"'))
    for chunks in (
        [_openai_chunk("Once"), _openai_chunk(finish_reason="length"), _openai_usage_chunk()],
        [_openai_chunk(tool_calls=[delta]), _openai_chunk(finish_reason="length"), _openai_usage_chunk()],
    ):
        responses = list(_openai(monkeypatch, iter(chunks)).stream("hi"))
        assert responses[-1].finish_reason == "length"
        assert responses[-1].usage.input_tokens == 10
        assert sum(len(r.tool_calls) for r in responses) == (1 if chunks[0].choices[0].delta.tool_calls else 0)


def test_openai_stream_without_usage_chunk_still_reports_length(monkeypatch):
    chunks = [_openai_chunk("Once"), _openai_chunk(finish_reason="length")]
    responses = list(_openai(monkeypatch, iter(chunks)).stream("hi"))
    assert responses[-1].finish_reason == "length"
