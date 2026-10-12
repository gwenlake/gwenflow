import asyncio
import json
import os
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from gwenflow.agents.agent import FINAL_RESULT_TOOL, Agent
from gwenflow.llms.base import ChatBase
from gwenflow.tools import BaseTool
from gwenflow.types import ModelResponse, RequestUsage, TextContent, ToolCall

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class AddTool(BaseTool):
    name: str = "add"
    description: str = "Adds two numbers"

    def _run(self, a: int, b: int) -> int:
        return a + b


def _mock_llm(responses):
    llm = MagicMock()
    llm.get_context_size.return_value = 96000
    llm.response_format = None
    llm.tools = []
    llm.tool_choice = None
    llm.invoke.side_effect = responses
    return llm


def _response(content="done", tool_calls=None):
    parts = []
    if content:
        parts.append(TextContent(content=content))
    if tool_calls:
        parts.extend(tool_calls)
    return ModelResponse(
        parts=parts,
        finish_reason="stop" if not tool_calls else "tool_calls",
        usage=RequestUsage(input_tokens=10, output_tokens=5),
    )


def skip_if_no_real_api_key():
    key = os.environ.get("OPENAI_API_KEY")
    return pytest.mark.skipif(key in [None, "", "test"], reason="OPENAI_API_KEY missing or fake value.")


# ---------------------------------------------------------------------------
# __post_init__
# ---------------------------------------------------------------------------


def test_agent_generates_id():
    agent = Agent(llm=_mock_llm([]))
    assert agent.id is not None


def test_agent_two_instances_have_different_ids():
    a1 = Agent(llm=_mock_llm([]))
    a2 = Agent(llm=_mock_llm([]))
    assert a1.id != a2.id


def test_agent_default_llm_is_openai():
    os.environ.setdefault("OPENAI_API_KEY", "fake")
    from gwenflow.llms.openai import ChatOpenAI

    agent = Agent()
    assert isinstance(agent.llm, ChatOpenAI)
    assert agent.llm.model == "gpt-5-mini"


def test_agent_history_initialized():
    agent = Agent(llm=_mock_llm([]))
    assert agent.history is not None


# ---------------------------------------------------------------------------
# get_system_prompt
# ---------------------------------------------------------------------------


def test_system_prompt_field_takes_priority():
    agent = Agent(system_prompt="You are a pirate.", llm=_mock_llm([]))
    assert agent.get_system_prompt(task="hello") == "You are a pirate."


def test_system_prompt_from_string_instructions():
    agent = Agent(instructions="Be concise.", llm=_mock_llm([]))
    prompt = agent.get_system_prompt(task="hello")
    assert "Be concise." in prompt


def test_system_prompt_from_list_instructions():
    agent = Agent(instructions=["Be concise.", "Be helpful."], llm=_mock_llm([]))
    prompt = agent.get_system_prompt(task="hello")
    assert "Be concise." in prompt
    assert "Be helpful." in prompt


def test_system_prompt_includes_response_model_schema():
    class MyOutput(BaseModel):
        answer: str

    agent = Agent(response_model=MyOutput, llm=_mock_llm([]))
    prompt = agent.get_system_prompt(task="hello")
    assert "answer" in prompt


def test_system_prompt_empty_without_instructions():
    agent = Agent(llm=_mock_llm([]))
    assert agent.get_system_prompt(task="hello") == ""


# ---------------------------------------------------------------------------
# _format_context
# ---------------------------------------------------------------------------


def test_format_context_string():
    agent = Agent(llm=_mock_llm([]))
    result = agent._format_context("some context")
    assert "<context>" in result
    assert "some context" in result


def test_format_context_dict():
    agent = Agent(llm=_mock_llm([]))
    result = agent._format_context({"background": "info here"})
    assert "<background>" in result
    assert "info here" in result


def test_format_context_none_returns_empty():
    agent = Agent(llm=_mock_llm([]))
    assert agent._format_context(None) == ""


# ---------------------------------------------------------------------------
# _validate_final_response
# ---------------------------------------------------------------------------


def test_validate_no_response_model_always_ok():
    agent = Agent(llm=_mock_llm([]))
    ok, data, err = agent._validate_final_response("anything")
    assert ok is True
    assert data == "anything"
    assert err is None


def test_validate_valid_pydantic_response():
    class MyOutput(BaseModel):
        answer: str

    agent = Agent(response_model=MyOutput, llm=_mock_llm([]))
    ok, data, err = agent._validate_final_response('{"answer": "yes"}')
    assert ok is True
    assert isinstance(data, MyOutput)
    assert data.answer == "yes"


def test_validate_invalid_pydantic_response():
    class MyOutput(BaseModel):
        answer: str

    agent = Agent(response_model=MyOutput, llm=_mock_llm([]))
    ok, data, err = agent._validate_final_response('{"wrong_field": "oops"}')
    assert ok is False
    assert err is not None


def test_validate_invalid_json():
    class MyOutput(BaseModel):
        answer: str

    agent = Agent(response_model=MyOutput, llm=_mock_llm([]))
    ok, data, err = agent._validate_final_response("not json at all")
    assert ok is False


# ---------------------------------------------------------------------------
# Agent.run — mocked LLM
# ---------------------------------------------------------------------------


def test_agent_run_simple_response():
    llm = _mock_llm([_response("The answer is 42")])
    agent = Agent(llm=llm)
    result = agent.run("What is 6 * 7?")

    assert result.content == "The answer is 42"
    assert result.finish_reason == "stop"


def test_agent_run_accumulates_usage():
    llm = _mock_llm([_response("hello")])
    agent = Agent(llm=llm)
    result = agent.run("hi")

    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 5


def test_agent_run_with_tool_call():
    tool_call = ToolCall(id="tc_1", name="add", arguments='{"a": 2, "b": 3}')
    llm = _mock_llm([_response(tool_calls=[tool_call]), _response("The result is 5")])
    agent = Agent(tools=[AddTool()], llm=llm)
    result = agent.run("Add 2 and 3")

    assert result.content == "The result is 5"


def test_agent_run_tool_call_increments_usage():
    tool_call = ToolCall(id="tc_1", name="add", arguments='{"a": 2, "b": 3}')
    llm = _mock_llm([_response(tool_calls=[tool_call]), _response("The result is 5")])
    agent = Agent(tools=[AddTool()], llm=llm)
    result = agent.run("Add 2 and 3")

    assert result.usage.tool_calls == 1
    assert result.usage.requests == 2


def test_agent_run_with_context_string():
    llm = _mock_llm([_response("answered")])
    agent = Agent(llm=llm)
    result = agent.run("question", context="some background")

    assert result.content == "answered"


def test_agent_run_respects_max_turns():
    tool_call = ToolCall(id="tc_1", name="add", arguments='{"a": 1, "b": 1}')
    # Always returns a tool call — agent should stop after max_turns
    llm = _mock_llm([_response(tool_calls=[tool_call])] * 10)
    agent = Agent(tools=[AddTool()], llm=llm, max_turns=3)
    result = agent.run("loop forever")

    assert result.finish_reason == "max_turns"
    assert result.content is None
    assert llm.invoke.call_count <= 3


# ---------------------------------------------------------------------------
# execute_tool_calls
# ---------------------------------------------------------------------------


def test_execute_tool_calls_returns_result():
    tool_call = ToolCall(id="tc_1", name="add", arguments='{"a": 2, "b": 3}')
    agent = Agent(tools=[AddTool()], llm=_mock_llm([]))
    messages = agent.execute_tool_calls([tool_call])

    assert len(messages) == 1
    assert messages[0].role == "tool"
    assert "5" in messages[0].content


def test_execute_tool_calls_unknown_tool():
    tool_call = ToolCall(id="tc_1", name="nonexistent", arguments="{}")
    agent = Agent(tools=[AddTool()], llm=_mock_llm([]))
    messages = agent.execute_tool_calls([tool_call])

    assert len(messages) == 1
    assert "does not exist" in messages[0].content


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


@pytest.mark.integration
@skip_if_no_real_api_key()
def test_agent_run_real_api():
    agent = Agent()
    result = agent.run("Reply with exactly one word: hello")

    assert result.content is not None
    assert result.finish_reason == "stop"
    assert result.usage.input_tokens > 0


# ---------------------------------------------------------------------------
# Memory budget: tool schemas
# ---------------------------------------------------------------------------


def test_tools_token_count_reserves_schema_budget():
    agent = Agent(name="a", instructions="Be terse.", llm=_mock_llm([]), tools=[AddTool()])
    assert agent._tools_token_count() > 0


def test_tools_token_count_is_zero_without_tools():
    agent = Agent(name="a", instructions="Be terse.", llm=_mock_llm([]))
    assert agent._tools_token_count() == 0


def test_prepare_history_sets_reserved_tokens():
    agent = Agent(name="a", instructions="Be terse.", llm=_mock_llm([]), tools=[AddTool()])
    assert agent.history.reserved_tokens == 0
    agent._prepare_history(task="do something")
    assert agent.history.reserved_tokens == agent._tools_token_count() > 0
    assert agent.history.system_prompt


# ---------------------------------------------------------------------------
# Memory budget: end-to-end tool loop
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class BigSearchTool(BaseTool):
    name: str = "search"
    description: str = "Search the web for a query and return results."

    def _run(self, query: str) -> str:
        """Search the web.

        Args:
            query: what to look for.
        """
        return f"RESULTS for {query}: " + "lorem ipsum dolor sit amet " * 60


def _capturing_llm(context_size, tool_turns):
    """LLM that requests a tool call `tool_turns` times, recording each prompt."""
    captured = []

    def invoke(input=None, **kwargs):
        captured.append(input)
        n = len(captured)
        if n <= tool_turns:
            return ModelResponse(
                parts=[ToolCall(id=f"call_{n}", name="search", arguments='{"query": "market ' + str(n) + '"}')],
                finish_reason="tool_calls",
                usage=RequestUsage(input_tokens=10, output_tokens=5),
            )
        return ModelResponse(
            parts=[TextContent(content="final report")],
            finish_reason="stop",
            usage=RequestUsage(input_tokens=10, output_tokens=5),
        )

    llm = _mock_llm([])
    llm.get_context_size.return_value = context_size
    llm.invoke.side_effect = invoke
    # A bare MagicMock returns an empty iterable here, so nothing would ever
    # reach memory and the test would pass vacuously.
    llm.input_to_message_list.side_effect = ChatBase.input_to_message_list.__get__(llm)
    llm.get_thinking_parts.return_value = None
    return llm, captured


def test_tool_loop_never_sends_a_task_less_prompt():
    """Every turn must carry the task, however long the tool loop runs.

    Regression: once a tool loop overflowed the budget, every later turn was
    sent to the model with nothing but a system prompt — the agent silently
    kept running with no task and no tool results.
    """
    task = "Analyse the French cloud market and write a report."
    llm, captured = _capturing_llm(context_size=1200, tool_turns=6)
    agent = Agent(name="analyst", instructions="Be terse.", llm=llm, tools=[BigSearchTool()])

    agent.run(task)

    assert len(captured) == 7
    for turn, messages in enumerate(captured, 1):
        roles = [m["role"] for m in messages]
        non_system = [r for r in roles if r != "system"]
        assert non_system, f"turn {turn} was sent with only a system prompt: {roles}"
        assert non_system[0] != "tool", f"turn {turn} opens on an orphan tool response: {roles}"
        assert any(
            task[:20] in str(m.get("content") or "") for m in messages
        ), f"turn {turn} lost the user task: {roles}"


def test_tool_loop_window_stays_bounded():
    """The window must still slide — keeping everything would defeat the buffer."""
    llm, captured = _capturing_llm(context_size=1200, tool_turns=6)
    agent = Agent(name="analyst", instructions="Be terse.", llm=llm, tools=[BigSearchTool()])

    agent.run("Analyse the French cloud market.")

    # Without pruning the last turn would carry 1 user + 6*(assistant+tool) = 13 messages
    assert len(captured[-1]) < 13


def test_streaming_tool_loop_never_sends_a_task_less_prompt():
    """Same regression on the streaming path.

    The run didn't crash, it just started talking to a model that had been
    given no task.
    """
    task = "Analyse the French cloud market and write a report."
    captured = []

    def stream(input=None, **kwargs):
        captured.append(input)
        n = len(captured)
        if n <= 6:
            yield ModelResponse(
                parts=[ToolCall(id=f"call_{n}", name="search", arguments='{"query": "market ' + str(n) + '"}')],
                finish_reason="tool_calls",
                usage=RequestUsage(input_tokens=10, output_tokens=5),
            )
        else:
            for word in ["final ", "report"]:
                yield ModelResponse(parts=[TextContent(content=word)], usage=RequestUsage(output_tokens=1))

    llm = _mock_llm([])
    llm.get_context_size.return_value = 1200
    llm.stream.side_effect = stream
    llm.input_to_message_list.side_effect = ChatBase.input_to_message_list.__get__(llm)
    llm.get_thinking_parts.return_value = None

    agent = Agent(name="analyst", instructions="Be terse.", llm=llm, tools=[BigSearchTool()])
    streamed = "".join(
        e.content for e in agent.run_stream(task) if type(e).__name__ == "AgentEventContent" and e.content
    )

    assert streamed == "final report"
    for turn, messages in enumerate(captured, 1):
        non_system = [m["role"] for m in messages if m["role"] != "system"]
        assert non_system, f"turn {turn} was streamed with only a system prompt"
        assert any(task[:20] in str(m.get("content") or "") for m in messages), f"turn {turn} lost the user task"


# ---------------------------------------------------------------------------
# response_model + tools: response_format deferred until no tool is needed
# ---------------------------------------------------------------------------


class _Answer(BaseModel):
    answer: str


def _record_calls(agent, responses):
    """Record response_format / tool_choice as the agent's LLM sees them at each invoke.

    The agent works on a private copy of the llm (see __post_init__), so the
    attributes must be read from `agent.llm`, not from the mock passed in.
    """
    calls = []

    def _invoke(input):
        calls.append({"response_format": agent.llm.response_format, "tool_choice": agent.llm.tool_choice})
        return responses.pop(0)

    agent.llm.invoke.side_effect = _invoke
    return calls


def test_response_format_sent_every_turn_without_tools():
    agent = Agent(response_model=_Answer, llm=_mock_llm([]))
    calls = _record_calls(agent, [_response('{"answer": "yes"}')])
    result = agent.run("hi")

    assert result.parsed == _Answer(answer="yes")
    assert calls[0]["response_format"] is _Answer


def test_response_format_withheld_while_tools_are_offered():
    tool_call = ToolCall(id="c1", name="add", arguments='{"a": 1, "b": 2}')
    agent = Agent(response_model=_Answer, llm=_mock_llm([]), tools=[AddTool()])
    calls = _record_calls(
        agent,
        [
            _response(content=None, tool_calls=[tool_call]),
            _response('{"answer": "3"}'),
        ],
    )
    result = agent.run("1 + 2 ?")

    assert result.parsed == _Answer(answer="3")
    assert result.usage.tool_calls == 1
    assert [c["response_format"] for c in calls] == [None, None]
    assert [c["tool_choice"] for c in calls] == [None, None]


def test_response_format_enforced_on_validation_retry_with_tools():
    tool_call = ToolCall(id="c1", name="add", arguments='{"a": 1, "b": 2}')
    agent = Agent(response_model=_Answer, llm=_mock_llm([]), tools=[AddTool()])
    calls = _record_calls(
        agent,
        [
            _response(content=None, tool_calls=[tool_call]),
            _response("The result is 3."),  # prose: fails validation
            _response('{"answer": "3"}'),  # structured retry
        ],
    )
    result = agent.run("1 + 2 ?")

    assert result.parsed == _Answer(answer="3")
    assert len(calls) == 3
    assert calls[1]["response_format"] is None
    assert calls[2]["response_format"] is _Answer
    assert calls[2]["tool_choice"] == "none"


def test_tool_choice_required_then_deferred_structured_output():
    tool_call = ToolCall(id="c1", name="add", arguments='{"a": 1, "b": 2}')
    agent = Agent(response_model=_Answer, llm=_mock_llm([]), tools=[AddTool()], tool_choice="required")
    calls = _record_calls(
        agent,
        [
            _response(content=None, tool_calls=[tool_call]),
            _response('{"answer": "3"}'),
        ],
    )
    result = agent.run("1 + 2 ?")

    assert result.parsed == _Answer(answer="3")
    assert [c["tool_choice"] for c in calls] == ["required", "auto"]
    assert [c["response_format"] for c in calls] == [None, None]


# ---------------------------------------------------------------------------
# response_model + tools: the answer arrives as a `final_result` tool call
# ---------------------------------------------------------------------------


def _last_assistant(agent):
    return [m for m in agent.history.get() if m.role == "assistant"][-1]


def test_final_result_tool_exposed_only_with_tools():
    with_tools = Agent(response_model=_Answer, llm=_mock_llm([]), tools=[AddTool()])
    names = [t.name for t in with_tools.llm.tools]
    assert names == ["add", FINAL_RESULT_TOOL]
    final = with_tools.llm.tools[-1]
    assert final.parameters == _Answer.model_json_schema()
    assert final.description

    without_tools = Agent(response_model=_Answer, llm=_mock_llm([]))
    assert without_tools.llm.tools is None

    no_schema = Agent(llm=_mock_llm([]), tools=[AddTool()])
    assert [t.name for t in no_schema.llm.tools] == ["add"]


def test_final_result_tool_added_by_tool_decorator():
    agent = Agent(response_model=_Answer, llm=_mock_llm([]))

    @agent.tool
    def ping() -> str:
        """Ping."""
        return "pong"

    assert [t.name for t in agent.llm.tools] == ["ping", FINAL_RESULT_TOOL]


def test_final_result_call_ends_the_run():
    """pydantic-ai style: the model ends its tool loop by calling final_result."""
    add = ToolCall(id="c1", name="add", arguments='{"a": 1, "b": 2}')
    final = ToolCall(id="c2", name=FINAL_RESULT_TOOL, arguments='{"answer": "3"}')
    agent = Agent(response_model=_Answer, llm=_mock_llm([]), tools=[AddTool()])
    calls = _record_calls(
        agent, [_response(content=None, tool_calls=[add]), _response(content=None, tool_calls=[final])]
    )
    result = agent.run("1 + 2 ?")

    assert result.parsed == _Answer(answer="3")
    assert result.finish_reason == "stop"
    assert result.usage.tool_calls == 1
    assert len(calls) == 2
    assert [c["response_format"] for c in calls] == [None, None]
    last = _last_assistant(agent)
    assert not last.tool_calls
    assert json.loads(last.content) == {"answer": "3"}


def test_final_result_with_invalid_args_gets_validation_feedback():
    bad = ToolCall(id="c1", name=FINAL_RESULT_TOOL, arguments='{"foo": 1}')
    good = ToolCall(id="c2", name=FINAL_RESULT_TOOL, arguments='{"answer": "ok"}')
    agent = Agent(response_model=_Answer, llm=_mock_llm([]), tools=[AddTool()])
    calls = _record_calls(
        agent, [_response(content=None, tool_calls=[bad]), _response(content=None, tool_calls=[good])]
    )
    result = agent.run("hi")

    assert result.parsed == _Answer(answer="ok")
    assert len(calls) == 2
    tool_msg = [m for m in agent.history.get() if m.role == "tool"][0]
    assert tool_msg.tool_call_id == "c1"
    assert "failed validation" in tool_msg.content
    assert "answer" in tool_msg.content  # pydantic names the missing field


def test_final_result_wins_over_other_tool_calls_in_the_same_turn():
    add = ToolCall(id="c1", name="add", arguments='{"a": 1, "b": 2}')
    final = ToolCall(id="c2", name=FINAL_RESULT_TOOL, arguments='{"answer": "3"}')
    agent = Agent(response_model=_Answer, llm=_mock_llm([]), tools=[AddTool()])
    calls = _record_calls(agent, [_response(content=None, tool_calls=[add, final])])
    result = agent.run("1 + 2 ?")

    assert result.parsed == _Answer(answer="3")
    assert result.usage.tool_calls == 0  # `add` was not executed
    assert len(calls) == 1
    assert not _last_assistant(agent).tool_calls


def test_schema_called_by_its_own_name_is_the_final_answer():
    """Qwen/Llama sometimes 'call' the schema title (`Answer`) instead of final_result."""
    add = ToolCall(id="c1", name="add", arguments='{"a": 1, "b": 2}')
    fake = ToolCall(id="c2", name="Answer", arguments='{"answer": "3"}')
    agent = Agent(response_model=_Answer, llm=_mock_llm([]), tools=[AddTool()])
    calls = _record_calls(
        agent, [_response(content=None, tool_calls=[add]), _response(content=None, tool_calls=[fake])]
    )
    result = agent.run("1 + 2 ?")

    assert result.parsed == _Answer(answer="3")
    assert result.usage.tool_calls == 1
    assert len(calls) == 2
    assert not _last_assistant(agent).tool_calls


def test_answer_tool_call_without_tools_is_absorbed_too():
    fake = ToolCall(id="c1", name="Answer", arguments='{"answer": "yes"}')
    agent = Agent(response_model=_Answer, llm=_mock_llm([]))
    _record_calls(agent, [_response(content=None, tool_calls=[fake])])
    result = agent.run("hi")

    assert result.parsed == _Answer(answer="yes")
    assert result.finish_reason == "stop"


def test_unknown_tool_call_with_invalid_args_points_to_final_result():
    bogus = ToolCall(id="c1", name="Answer", arguments='{"foo": 1}')
    agent = Agent(response_model=_Answer, llm=_mock_llm([]), tools=[AddTool()])
    calls = _record_calls(agent, [_response(content=None, tool_calls=[bogus]), _response('{"answer": "ok"}')])
    result = agent.run("hi")

    assert result.parsed == _Answer(answer="ok")
    assert len(calls) == 2
    tool_msg = [m for m in agent.history.get() if m.role == "tool"][0]
    assert "does not exist" in tool_msg.content
    assert FINAL_RESULT_TOOL in tool_msg.content


def test_real_tool_call_is_never_mistaken_for_an_answer():
    """A genuine tool whose arguments happen to fit the schema still runs."""

    @dataclass(kw_only=True)
    class EchoTool(BaseTool):
        name: str = "echo"
        description: str = "Echoes"

        def _run(self, answer: str) -> str:
            return answer.upper()

    echo = ToolCall(id="c1", name="echo", arguments='{"answer": "hi"}')
    agent = Agent(response_model=_Answer, llm=_mock_llm([]), tools=[EchoTool()])
    _record_calls(agent, [_response(content=None, tool_calls=[echo]), _response('{"answer": "HI"}')])
    result = agent.run("hi")

    assert result.parsed == _Answer(answer="HI")
    assert result.usage.tool_calls == 1


def test_final_result_with_lax_typed_arguments():
    """The int field arrives as a string and pydantic coerces it."""

    class _Total(BaseModel):
        total_eur: int

    final = ToolCall(id="c1", name=FINAL_RESULT_TOOL, arguments='{"total_eur": "7391"}')
    agent = Agent(response_model=_Total, llm=_mock_llm([]), tools=[AddTool()])
    calls = _record_calls(agent, [_response(content=None, tool_calls=[final])])
    result = agent.run("total?")

    assert result.parsed == _Total(total_eur=7391)
    assert len(calls) == 1
    assert json.loads(_last_assistant(agent).content) == {"total_eur": 7391}


def test_final_result_absorbed_on_streaming_paths():
    add = ToolCall(id="c1", name="add", arguments='{"a": 1, "b": 2}')
    final = ToolCall(id="c2", name=FINAL_RESULT_TOOL, arguments='{"answer": "3"}')

    def stream(input=None, **kwargs):
        tc = add if len(input) <= 2 else final  # system+user -> tool turn; afterwards -> final
        yield ModelResponse(parts=[tc], finish_reason="tool_calls", usage=RequestUsage(output_tokens=1))

    async def astream(input=None, **kwargs):
        for chunk in stream(input):
            yield chunk

    llm = _mock_llm([])
    llm.stream.side_effect = stream
    llm.astream.side_effect = astream
    llm.input_to_message_list.side_effect = ChatBase.input_to_message_list.__get__(llm)
    llm.get_thinking_parts.return_value = None

    agent = Agent(response_model=_Answer, llm=llm, tools=[AddTool()])
    events = list(agent.run_stream("1 + 2 ?"))
    streamed = "".join(e.content for e in events if type(e).__name__ == "AgentEventContent" and e.content)
    assert json.loads(streamed) == {"answer": "3"}
    assert not _last_assistant(agent).tool_calls

    async def go():
        agent = Agent(response_model=_Answer, llm=llm, tools=[AddTool()])
        out = [e async for e in agent.arun_stream("1 + 2 ?")]
        return agent, out

    agent2, events2 = asyncio.run(go())
    streamed2 = "".join(e.content for e in events2 if type(e).__name__ == "AgentEventContent" and e.content)
    assert json.loads(streamed2) == {"answer": "3"}
    assert not _last_assistant(agent2).tool_calls
