"""Unit tests for ChatGoogle on the native google-genai SDK.

No live API calls: we mock the SDK client and verify message translation
(role mapping, system_instruction split-out), Part construction for
multi-modal input, ThinkingConfig wiring, tool declarations, and response
parsing including thought signatures.
"""

import base64
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gwenflow.llms.google import ChatGoogle
from gwenflow.tools import Tool
from gwenflow.types import (
    AudioContent,
    FileContent,
    ImageContent,
    Message,
    ModelResponse,
    TextContent,
    ThinkingContent,
)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_construct_default_model():
    llm = ChatGoogle(api_key="test")
    assert llm.model == "gemini-2.5-flash"


def test_construct_missing_key_raises(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
        ChatGoogle()._get_client_params()


# ---------------------------------------------------------------------------
# Message formatting
# ---------------------------------------------------------------------------


def test_format_messages_text_user_to_user_role():
    llm = ChatGoogle(api_key="test")
    formatted = llm._format_messages([Message(role="user", content="hi")])
    assert formatted[0].role == "user"
    assert formatted[0].parts[0].text == "hi"


def test_format_messages_assistant_to_model_role():
    llm = ChatGoogle(api_key="test")
    formatted = llm._format_messages([Message(role="assistant", content="ok")])
    assert formatted[0].role == "model"


def test_format_messages_system_instruction_extracted_not_in_contents():
    llm = ChatGoogle(api_key="test")
    formatted = llm._format_messages([
        Message(role="system", content="be terse"),
        Message(role="user", content="hi"),
    ])
    assert llm.system_prompt == "be terse"
    assert len(formatted) == 1
    assert formatted[0].role == "user"


def test_format_image_url_to_file_data():
    llm = ChatGoogle(api_key="test")
    m = Message(role="user", content=[ImageContent.from_url("https://x/i.jpg")])
    formatted = llm._format_messages([m])
    part = formatted[0].parts[0]
    assert part.file_data.file_uri == "https://x/i.jpg"


def test_format_image_base64_to_inline_data():
    llm = ChatGoogle(api_key="test")
    img = ImageContent.from_bytes(b"hello", media_type="image/png")
    m = Message(role="user", content=[img])
    formatted = llm._format_messages([m])
    part = formatted[0].parts[0]
    assert part.inline_data.mime_type == "image/png"
    assert part.inline_data.data == b"hello"


def test_format_audio_to_inline_data():
    llm = ChatGoogle(api_key="test")
    a = AudioContent.from_bytes(b"wavbytes", format="wav")
    m = Message(role="user", content=[a])
    formatted = llm._format_messages([m])
    part = formatted[0].parts[0]
    assert part.inline_data.mime_type == "audio/wav"
    assert part.inline_data.data == b"wavbytes"


def test_format_file_url_to_file_data():
    llm = ChatGoogle(api_key="test")
    m = Message(role="user", content=[FileContent.from_url("https://x/y.pdf")])
    formatted = llm._format_messages([m])
    part = formatted[0].parts[0]
    assert part.file_data.file_uri == "https://x/y.pdf"
    assert part.file_data.mime_type == "application/pdf"


def test_format_assistant_tool_call_becomes_function_call_part():
    llm = ChatGoogle(api_key="test")
    m = Message(
        role="assistant",
        content="",
        tool_calls=[{"id": "t1", "type": "function", "function": {"name": "add", "arguments": '{"x": 1}'}}],
    )
    formatted = llm._format_messages([m])
    part = formatted[0].parts[0]
    assert part.function_call.name == "add"
    assert part.function_call.args == {"x": 1}


def test_format_tool_response_becomes_function_response_part():
    llm = ChatGoogle(api_key="test")
    m = Message(role="tool", tool_call_id="t1", name="add", content='{"result": 3}')
    formatted = llm._format_messages([m])
    assert formatted[0].role == "user"  # Gemini puts tool results under user
    part = formatted[0].parts[0]
    assert part.function_response.name == "add"
    assert part.function_response.response == {"result": 3}


def test_thinking_parts_echo_with_signature_decoded():
    """Replayed thinking_parts must have thought=True and the signature decoded to bytes."""
    llm = ChatGoogle(api_key="test")
    m = Message(
        role="assistant",
        content="answer",
        thinking_parts=[ThinkingContent(content="reasoning", extra={"thought_signature": "YWJj"})],
    )
    formatted = llm._format_messages([m])
    part = formatted[0].parts[0]
    assert part.thought is True
    assert part.thought_signature == b"abc"


# ---------------------------------------------------------------------------
# Config building
# ---------------------------------------------------------------------------


def test_config_includes_temperature():
    llm = ChatGoogle(api_key="test", temperature=0.7, top_p=0.9, max_output_tokens=200)
    cfg = llm._build_config()
    assert cfg.temperature == 0.7
    assert cfg.top_p == 0.9
    assert cfg.max_output_tokens == 200


def test_config_thinking_block():
    llm = ChatGoogle(api_key="test", thinking={"include_thoughts": True, "thinking_budget": 512})
    cfg = llm._build_config()
    assert cfg.thinking_config.include_thoughts is True
    assert cfg.thinking_config.thinking_budget == 512


def test_config_tools_function_declaration():
    def add(x: int, y: int) -> int:
        """Add."""
        return x + y

    llm = ChatGoogle(api_key="test", tools=[Tool(add)])
    cfg = llm._build_config()
    assert cfg.tools is not None
    decl = cfg.tools[0].function_declarations[0]
    assert decl.name == "add"


def test_config_response_format_json_object():
    llm = ChatGoogle(api_key="test", response_format={"type": "json_object"})
    cfg = llm._build_config()
    assert cfg.response_mime_type == "application/json"


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _fake_part(*, text=None, thought=False, thought_signature=None, function_call=None, inline_data=None):
    return SimpleNamespace(
        text=text,
        thought=thought,
        thought_signature=thought_signature,
        function_call=function_call,
        inline_data=inline_data,
    )


def _fake_candidate(parts, finish_reason="STOP"):
    content = SimpleNamespace(parts=parts, role="model")
    return SimpleNamespace(content=content, finish_reason=finish_reason)


def _fake_response(candidates, usage=None):
    return SimpleNamespace(
        candidates=candidates,
        usage_metadata=usage
        or SimpleNamespace(
            prompt_token_count=10,
            candidates_token_count=5,
            cached_content_token_count=0,
            thoughts_token_count=0,
            total_token_count=15,
        ),
    )


def test_parse_response_text_only():
    llm = ChatGoogle(api_key="test")
    resp = llm._parse_response(_fake_response([_fake_candidate([_fake_part(text="Hello.")])]))
    assert resp.text == "Hello."
    assert resp.usage.input_tokens == 10
    assert resp.usage.output_tokens == 5


def test_parse_response_thinking_captured_with_signature():
    sig = b"\xab\xcd"
    parts = [
        _fake_part(text="I'm thinking...", thought=True, thought_signature=sig),
        _fake_part(text="42"),
    ]
    llm = ChatGoogle(api_key="test")
    resp = llm._parse_response(_fake_response([_fake_candidate(parts)]))
    thinking_parts = [p for p in resp.parts if isinstance(p, ThinkingContent)]
    assert thinking_parts[0].content == "I'm thinking..."
    assert thinking_parts[0].extra["thought_signature"] == base64.b64encode(sig).decode("ascii")
    assert resp.text == "42"


def test_parse_response_function_call_becomes_tool_call():
    fc = SimpleNamespace(name="add", args={"x": 1}, id="call_1")
    parts = [_fake_part(function_call=fc)]
    llm = ChatGoogle(api_key="test")
    resp = llm._parse_response(_fake_response([_fake_candidate(parts)]))
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "add"
    assert resp.finish_reason == "tool_calls"


def test_parse_response_inline_image_becomes_image_content():
    inline = SimpleNamespace(data=b"\x89PNG...", mime_type="image/png")
    parts = [_fake_part(inline_data=inline)]
    llm = ChatGoogle(api_key="test")
    resp = llm._parse_response(_fake_response([_fake_candidate(parts)]))
    assert len(resp.images) == 1
    assert resp.images[0].media_type == "image/png"
    assert resp.images[0].data == base64.b64encode(b"\x89PNG...").decode("ascii")


def test_parse_response_usage_includes_thinking_tokens():
    usage = SimpleNamespace(
        prompt_token_count=10,
        candidates_token_count=5,
        cached_content_token_count=2,
        thoughts_token_count=20,
        total_token_count=35,
    )
    llm = ChatGoogle(api_key="test")
    resp = llm._parse_response(_fake_response([_fake_candidate([_fake_part(text="x")])], usage=usage))
    assert resp.usage.cache_read_tokens == 2
    assert resp.usage.details.get("reasoning_tokens") == 20


# ---------------------------------------------------------------------------
# get_thinking_parts
# ---------------------------------------------------------------------------


def test_get_thinking_parts_returns_none_when_empty():
    llm = ChatGoogle(api_key="test")
    assert llm.get_thinking_parts(ModelResponse(parts=[TextContent(content="hi")])) is None


def test_get_thinking_parts_preserves_signature():
    llm = ChatGoogle(api_key="test")
    mr = ModelResponse(parts=[ThinkingContent(content="r", extra={"thought_signature": "abc"})])
    parts = llm.get_thinking_parts(mr)
    assert parts[0].extra["thought_signature"] == "abc"


# ---------------------------------------------------------------------------
# End-to-end mocked invoke
# ---------------------------------------------------------------------------


def test_invoke_returns_parsed_response_via_mocked_client():
    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = _fake_response(
        [_fake_candidate([_fake_part(text="Hello back.")])]
    )

    llm = ChatGoogle(api_key="test")
    llm.client = fake_client

    resp = llm.invoke("hi")
    assert resp.text == "Hello back."

    _, kwargs = fake_client.models.generate_content.call_args
    assert kwargs["model"] == "gemini-2.5-flash"
    # The contents must be Gemini Content objects, not raw dicts
    from google.genai import types as gt

    assert all(isinstance(c, gt.Content) for c in kwargs["contents"])
