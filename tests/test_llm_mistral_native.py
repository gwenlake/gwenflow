"""Unit tests for ChatMistral on the native mistralai SDK.

No live API calls: we mock the SDK client and verify message translation,
response parsing (including ThinkChunk from Magistral), tool wire shape, and
multi-modal input formatting.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gwenflow.llms.mistral import ChatMistral
from gwenflow.tools import Tool
from gwenflow.types import (
    ImageContent,
    Message,
    ModelResponse,
    TextContent,
    ThinkingContent,
)


# ---------------------------------------------------------------------------
# Construction and config
# ---------------------------------------------------------------------------


def test_construct_uses_default_model():
    llm = ChatMistral(api_key="test")
    assert llm.model == "mistral-small-2603"


def test_construct_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    with pytest.raises(ValueError, match="MISTRAL_API_KEY"):
        ChatMistral()._get_client_params()


# ---------------------------------------------------------------------------
# Message formatting → Mistral SDK objects
# ---------------------------------------------------------------------------


def test_format_messages_user_text():
    from mistralai.client.models import UserMessage

    llm = ChatMistral(api_key="test")
    formatted = llm._format_messages([Message(role="user", content="hello")])
    assert isinstance(formatted[0], UserMessage)
    assert formatted[0].content == "hello"


def test_format_messages_system_role():
    from mistralai.client.models import SystemMessage

    llm = ChatMistral(api_key="test")
    formatted = llm._format_messages([Message(role="system", content="be brief")])
    assert isinstance(formatted[0], SystemMessage)


def test_format_messages_assistant_with_tool_calls():
    from mistralai.client.models import AssistantMessage

    llm = ChatMistral(api_key="test")
    m = Message(
        role="assistant",
        content="ok",
        tool_calls=[{"id": "t1", "type": "function", "function": {"name": "foo", "arguments": "{}"}}],
    )
    formatted = llm._format_messages([m])
    assert isinstance(formatted[0], AssistantMessage)
    tc = formatted[0].tool_calls[0]
    # The Mistral SDK coerces our dict into its own ToolCall object
    name = tc["function"]["name"] if isinstance(tc, dict) else tc.function.name
    assert name == "foo"


def test_format_messages_tool_role():
    from mistralai.client.models import ToolMessage

    llm = ChatMistral(api_key="test")
    m = Message(role="tool", tool_call_id="t1", content="42", name="add")
    formatted = llm._format_messages([m])
    assert isinstance(formatted[0], ToolMessage)
    assert formatted[0].tool_call_id == "t1"


def test_format_image_url_chunk():
    from mistralai.client.models import ImageURLChunk

    llm = ChatMistral(api_key="test")
    m = Message(role="user", content=[ImageContent.from_url("https://x/i.jpg")])
    formatted = llm._format_messages([m])
    chunk = formatted[0].content[0]
    assert isinstance(chunk, ImageURLChunk)


def test_format_image_base64_becomes_data_uri():
    from mistralai.client.models import ImageURLChunk

    llm = ChatMistral(api_key="test")
    img = ImageContent.from_bytes(b"x", media_type="image/png")
    m = Message(role="user", content=[img])
    formatted = llm._format_messages([m])
    chunk = formatted[0].content[0]
    assert isinstance(chunk, ImageURLChunk)
    # image_url can be a string or an object with .url; both should embed data:
    url = chunk.image_url if isinstance(chunk.image_url, str) else chunk.image_url.url
    assert url.startswith("data:image/png;base64,")


def test_assistant_thinking_parts_become_think_chunks():
    from mistralai.client.models import ThinkChunk

    llm = ChatMistral(api_key="test")
    m = Message(
        role="assistant",
        content="answer",
        thinking_parts=[ThinkingContent(content="step 1"), ThinkingContent(content="step 2")],
    )
    formatted = llm._format_messages([m])
    chunks = formatted[0].content
    assert isinstance(chunks[0], ThinkChunk)
    assert isinstance(chunks[1], ThinkChunk)


# ---------------------------------------------------------------------------
# Tool wire shape
# ---------------------------------------------------------------------------


def test_tools_wire_format():
    def add(x: int, y: int) -> int:
        """Add two numbers."""
        return x + y

    llm = ChatMistral(api_key="test", tools=[Tool(add)])
    params = llm._model_params
    assert "tools" in params
    assert params["tools"][0] == {
        "type": "function",
        "function": {"name": "add", "description": params["tools"][0]["function"]["description"], "parameters": params["tools"][0]["function"]["parameters"]},
    }
    assert params["tool_choice"] == "auto"


# ---------------------------------------------------------------------------
# Response parsing (incl. ThinkChunk)
# ---------------------------------------------------------------------------


def _fake_mistral_response(content, *, tool_calls=None, finish_reason="stop"):
    msg = SimpleNamespace(role="assistant", content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(index=0, message=msg, finish_reason=finish_reason)
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    return SimpleNamespace(id="r1", choices=[choice], usage=usage, object=None, created=None, model="m")


def test_parse_response_text_only():
    llm = ChatMistral(api_key="test")
    resp = llm._parse_response(_fake_mistral_response("Hello world."))
    assert resp.text == "Hello world."
    assert resp.usage.input_tokens == 10
    assert resp.usage.output_tokens == 5


def test_parse_response_think_chunk_extracted():
    from mistralai.client.models import TextChunk, ThinkChunk

    content = [
        ThinkChunk(thinking=[{"type": "text", "text": "let me think..."}], closed=True),
        TextChunk(text="42"),
    ]
    llm = ChatMistral(api_key="test")
    resp = llm._parse_response(_fake_mistral_response(content))
    assert resp.thinking == "let me think..."
    assert resp.text == "42"


def test_parse_response_tool_calls():
    tool_calls = [
        SimpleNamespace(
            id="t1",
            type="function",
            function=SimpleNamespace(name="add", arguments='{"x": 1, "y": 2}'),
        )
    ]
    llm = ChatMistral(api_key="test")
    resp = llm._parse_response(_fake_mistral_response("", tool_calls=tool_calls, finish_reason="tool_calls"))
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "add"
    assert resp.finish_reason == "tool_calls"


# ---------------------------------------------------------------------------
# get_thinking_parts
# ---------------------------------------------------------------------------


def test_get_thinking_parts_returns_none_when_no_thinking():
    llm = ChatMistral(api_key="test")
    assert llm.get_thinking_parts(ModelResponse(parts=[TextContent(content="hi")])) is None


def test_get_thinking_parts_coalesces():
    llm = ChatMistral(api_key="test")
    mr = ModelResponse(parts=[ThinkingContent(content="a"), ThinkingContent(content="b")])
    parts = llm.get_thinking_parts(mr)
    assert len(parts) == 1
    assert parts[0].content == "ab"


# ---------------------------------------------------------------------------
# End-to-end mocked invoke
# ---------------------------------------------------------------------------


def test_invoke_calls_complete_and_returns_parsed_response():
    fake_client = MagicMock()
    fake_client.chat.complete.return_value = _fake_mistral_response("Hello back.")

    llm = ChatMistral(api_key="test")
    llm.client = fake_client

    resp = llm.invoke("hi")
    assert resp.text == "Hello back."

    # The Mistral SDK got actual SDK message objects, not raw dicts
    from mistralai.client.models import UserMessage

    _, kwargs = fake_client.chat.complete.call_args
    assert isinstance(kwargs["messages"][0], UserMessage)
    assert kwargs["model"] == "mistral-small-2603"
