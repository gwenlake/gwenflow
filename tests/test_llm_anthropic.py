import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gwenflow.llms.anthropic.chat import ChatAnthropic
from gwenflow.types import Message

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_anthropic_response(text="hello from claude", stop_reason="end_turn"):
    """Build a minimal object that looks like anthropic.types.Message."""
    content_block = SimpleNamespace(type="text", text=text)
    usage = SimpleNamespace(input_tokens=10, output_tokens=5)
    return SimpleNamespace(content=[content_block], stop_reason=stop_reason, usage=usage)


# ---------------------------------------------------------------------------
# Unit tests (mocked client)
# ---------------------------------------------------------------------------


def test_invoke_returns_content_with_mock(monkeypatch):
    fake_response = _make_anthropic_response("mocked claude response")
    fake_client = MagicMock()
    fake_client.messages.create.return_value = fake_response
    monkeypatch.setattr(ChatAnthropic, "get_client", lambda self: fake_client)

    llm = ChatAnthropic(api_key="fake-key")
    result = llm.invoke("Say hello")

    assert result.content == "mocked claude response"
    assert result.finish_reason == "stop"
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 5


def test_invoke_with_tool_calls_mock(monkeypatch):
    tool_block = SimpleNamespace(type="tool_use", id="tu_1", name="my_tool", input={"x": 1})
    usage = SimpleNamespace(input_tokens=20, output_tokens=8)
    fake_response = SimpleNamespace(content=[tool_block], stop_reason="tool_use", usage=usage)

    fake_client = MagicMock()
    fake_client.messages.create.return_value = fake_response
    monkeypatch.setattr(ChatAnthropic, "get_client", lambda self: fake_client)

    llm = ChatAnthropic(api_key="fake-key")
    result = llm.invoke("Use a tool")

    assert result.finish_reason == "tool_calls"
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.name == "my_tool"


def test_format_messages_extracts_system_prompt():
    llm = ChatAnthropic(api_key="fake-key")
    messages = [
        Message(role="system", content="You are helpful."),
        Message(role="user", content="Hi"),
    ]

    # Correction : on récupère juste la liste retournée
    formatted = llm._format_messages(messages)

    # Correction : on lit la propriété de l'instance
    assert llm.system_prompt == "You are helpful."
    assert len(formatted) == 1
    assert formatted[0]["role"] == "user"


def test_format_messages_tool_role():
    llm = ChatAnthropic(api_key="fake-key")
    messages = [
        Message(role="tool", content="result", tool_call_id="call_123"),
    ]

    # Correction : on récupère juste la liste retournée
    formatted = llm._format_messages(messages)

    assert formatted[0]["role"] == "user"
    assert formatted[0]["content"][0]["type"] == "tool_result"
    assert formatted[0]["content"][0]["tool_use_id"] == "call_123"


# ---------------------------------------------------------------------------
# Integration test (real API)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY missing")
def test_invoke_real_api():
    llm = ChatAnthropic()
    result = llm.invoke("Reply with exactly one word: hello")

    assert result.content is not None
    assert len(result.content) > 0
    assert result.usage is not None
    assert result.usage.input_tokens > 0


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY missing")
def test_stream_real_api():
    llm = ChatAnthropic()
    chunks = list(llm.stream("Reply with exactly one word: hello"))

    text_chunks = [c for c in chunks if c.content]
    assert len(text_chunks) > 0
    full_text = "".join(c.content for c in text_chunks)
    assert len(full_text) > 0
