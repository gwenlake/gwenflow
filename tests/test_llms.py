import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gwenflow.llms.models import MODELS
from gwenflow.llms.openai import ChatOpenAI
from gwenflow.types import Message


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_openai_response(text="hello", tool_calls=None):
    message = SimpleNamespace(role="assistant", content=text, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message, finish_reason="stop")
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    return SimpleNamespace(choices=[choice], usage=usage)


# ---------------------------------------------------------------------------
# MODELS
# ---------------------------------------------------------------------------


def test_known_model_context_window():
    assert MODELS["gpt-4o"]["context_window"] == 128000


def test_known_model_reasoning_false():
    assert MODELS["gpt-4o"]["reasoning"] is False


def test_known_model_reasoning_true():
    assert MODELS["o3"]["reasoning"] is True


def test_gpt41_variants_same_context():
    assert MODELS["gpt-4.1"]["context_window"] == MODELS["gpt-4.1-mini"]["context_window"]
    assert MODELS["gpt-4.1"]["context_window"] == MODELS["gpt-4.1-nano"]["context_window"]


def test_anthropic_haiku_versioned_matches_base():
    assert MODELS["claude-haiku-4-5"]["context_window"] == MODELS["claude-haiku-4-5-20251001"]["context_window"]


def test_mistral_default_model_present():
    assert "open-mistral-7b" in MODELS


# ---------------------------------------------------------------------------
# ChatBase utility methods
# ---------------------------------------------------------------------------


def test_get_context_window_known_model():
    llm = ChatOpenAI(model="gpt-4.1", api_key="fake")
    assert llm.get_context_size() == int(1047576 * 0.75)


def test_get_context_window_unknown_model():
    llm = ChatOpenAI(model="unknown-model-xyz", api_key="fake")
    assert llm.get_context_size() == int(128000 * 0.75)


def test_get_reasoning_model_true():
    llm = ChatOpenAI(model="o3", api_key="fake")
    assert llm.get_reasoning_model() is True


def test_get_reasoning_model_false():
    llm = ChatOpenAI(model="gpt-4o", api_key="fake")
    assert llm.get_reasoning_model() is False


def test_get_tool_names_and_map():
    from dataclasses import dataclass
    from gwenflow.tools import BaseTool

    @dataclass(kw_only=True)
    class DummyTool(BaseTool):
        name: str = "dummy"
        description: str = "A dummy tool"

        def _run(self, **kwargs):
            return "ok"

    tool = DummyTool()
    llm = ChatOpenAI(api_key="fake", tools=[tool])
    assert llm.get_tool_names() == ["dummy"]
    assert llm.get_tool_map() == {"dummy": tool}


# ---------------------------------------------------------------------------
# ChatOpenAI — mocked client
# ---------------------------------------------------------------------------


def test_openai_invoke_returns_content(monkeypatch):
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = _make_openai_response("hello from openai")
    monkeypatch.setattr(ChatOpenAI, "get_client", lambda self: fake_client)

    llm = ChatOpenAI(api_key="fake-key")
    result = llm.invoke("Say hello")

    assert result.content == "hello from openai"
    assert result.finish_reason == "stop"
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 5


def test_openai_invoke_with_tool_calls(monkeypatch):
    tool_call = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(name="my_tool", arguments='{"x": 1}'),
    )
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = _make_openai_response("", tool_calls=[tool_call])
    monkeypatch.setattr(ChatOpenAI, "get_client", lambda self: fake_client)

    llm = ChatOpenAI(api_key="fake-key")
    result = llm.invoke("Use a tool")

    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "my_tool"


def test_openai_invoke_with_message_list(monkeypatch):
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = _make_openai_response("pong")
    monkeypatch.setattr(ChatOpenAI, "get_client", lambda self: fake_client)

    llm = ChatOpenAI(api_key="fake-key")
    messages = [Message(role="user", content="ping")]
    result = llm.invoke(messages)

    assert result.content == "pong"


def test_openai_invoke_no_usage_returns_none(monkeypatch):
    response = _make_openai_response("ok")
    response = SimpleNamespace(choices=response.choices, usage=None)
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = response
    monkeypatch.setattr(ChatOpenAI, "get_client", lambda self: fake_client)

    llm = ChatOpenAI(api_key="fake-key")
    result = llm.invoke("hi")

    assert result.usage is None


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY missing")
def test_openai_invoke_real_api():
    llm = ChatOpenAI()
    result = llm.invoke("Reply with exactly one word: hello")

    assert result.content is not None
    assert len(result.content) > 0
    assert result.usage.input_tokens > 0
