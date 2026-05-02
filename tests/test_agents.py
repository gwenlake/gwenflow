import os
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from gwenflow.agents.agent import Agent
from gwenflow.tools import Tool
from gwenflow.types import Message, ModelResponse, RequestUsage
from gwenflow.types.response import TextPart, ToolCallPart


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class AddTool(Tool):
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
        parts.append(TextPart(content=content))
    if tool_calls:
        parts.extend(tool_calls)
    return ModelResponse(
        parts=parts,
        finish_reason="stop" if not tool_calls else "tool_calls",
        usage=RequestUsage(input_tokens=10, output_tokens=5),
    )


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
    tool_call = ToolCallPart(id="tc_1", function="add", arguments='{"a": 2, "b": 3}')
    llm = _mock_llm([_response(tool_calls=[tool_call]), _response("The result is 5")])
    agent = Agent(tools=[AddTool()], llm=llm)
    result = agent.run("Add 2 and 3")

    assert result.content == "The result is 5"


def test_agent_run_tool_call_increments_usage():
    tool_call = ToolCallPart(id="tc_1", function="add", arguments='{"a": 2, "b": 3}')
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
    tool_call = ToolCallPart(id="tc_1", function="add", arguments='{"a": 1, "b": 1}')
    # Always returns a tool call — agent should stop after max_turns
    llm = _mock_llm([_response(tool_calls=[tool_call])] * 10)
    agent = Agent(tools=[AddTool()], llm=llm, max_turns=3)
    result = agent.run("loop forever")

    assert result.finish_reason == "stop"
    assert llm.invoke.call_count <= 3


# ---------------------------------------------------------------------------
# execute_tool_calls
# ---------------------------------------------------------------------------


def test_execute_tool_calls_returns_result():
    tool_call = ToolCallPart(id="tc_1", function="add", arguments='{"a": 2, "b": 3}')
    agent = Agent(tools=[AddTool()], llm=_mock_llm([]))
    messages = agent.execute_tool_calls([tool_call])

    assert len(messages) == 1
    assert messages[0].role == "tool"
    assert "5" in messages[0].content


def test_execute_tool_calls_unknown_tool():
    tool_call = ToolCallPart(id="tc_1", function="nonexistent", arguments="{}")
    agent = Agent(tools=[AddTool()], llm=_mock_llm([]))
    messages = agent.execute_tool_calls([tool_call])

    assert len(messages) == 1
    assert "does not exist" in messages[0].content


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY missing")
def test_agent_run_real_api():
    agent = Agent()
    result = agent.run("Reply with exactly one word: hello")

    assert result.content is not None
    assert result.finish_reason == "stop"
    assert result.usage.input_tokens > 0
