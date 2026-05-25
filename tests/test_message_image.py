"""Unit tests for image input handling (Phase 1 multi-modal).

Covers:
- ImageContent constructors (from_url, from_bytes, from_path)
- Message.to_openai() translation to Chat-Completions wire shape
- ChatAnthropic._format_messages translation to Anthropic blocks
- JSON roundtrip rehydrates content parts back into dataclasses
- Raw provider-shaped dicts pass through unchanged
- Mocked end-to-end invoke with an image
"""

import base64
import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gwenflow.llms.anthropic import ChatAnthropic
from gwenflow.llms.openai import ChatOpenAI
from gwenflow.types import ImageContent, Message, TextContent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# 1x1 red PNG, base64.
RED_PIXEL_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg=="
)
RED_PIXEL_PNG = base64.b64decode(RED_PIXEL_PNG_B64)


@pytest.fixture
def red_pixel_path(tmp_path: Path) -> Path:
    p = tmp_path / "red.png"
    p.write_bytes(RED_PIXEL_PNG)
    return p


# ---------------------------------------------------------------------------
# ImageContent constructors
# ---------------------------------------------------------------------------


def test_image_content_from_url():
    img = ImageContent.from_url("https://example.com/cat.jpg")
    assert img.url == "https://example.com/cat.jpg"
    assert img.data is None
    assert img.kind == "image"


def test_image_content_from_url_with_detail():
    img = ImageContent.from_url("https://example.com/cat.jpg", detail="high")
    assert img.detail == "high"


def test_image_content_from_bytes():
    img = ImageContent.from_bytes(RED_PIXEL_PNG, media_type="image/png")
    assert img.data == RED_PIXEL_PNG_B64
    assert img.media_type == "image/png"
    assert img.url is None


def test_image_content_from_path(red_pixel_path: Path):
    img = ImageContent.from_path(red_pixel_path)
    assert img.data == RED_PIXEL_PNG_B64
    assert img.media_type == "image/png"


def test_image_content_from_path_default_mime_for_unknown_ext(tmp_path: Path):
    p = tmp_path / "blob.weird"
    p.write_bytes(b"x")
    img = ImageContent.from_path(p)
    assert img.media_type == "image/jpeg"  # falls back to default


# ---------------------------------------------------------------------------
# Message.to_openai() — OpenAI Chat Completions wire shape
# ---------------------------------------------------------------------------


def test_to_openai_string_content_unchanged():
    m = Message(role="user", content="hello")
    assert m.to_openai()["content"] == "hello"


def test_to_openai_text_only_list():
    m = Message(role="user", content=[TextContent(content="hi")])
    assert m.to_openai()["content"] == [{"type": "text", "text": "hi"}]


def test_to_openai_text_plus_image_url():
    m = Message(
        role="user",
        content=[
            TextContent(content="describe"),
            ImageContent.from_url("https://example.com/x.jpg"),
        ],
    )
    assert m.to_openai()["content"] == [
        {"type": "text", "text": "describe"},
        {"type": "image_url", "image_url": {"url": "https://example.com/x.jpg"}},
    ]


def test_to_openai_image_base64_becomes_data_uri():
    img = ImageContent.from_bytes(RED_PIXEL_PNG, media_type="image/png", detail="high")
    m = Message(role="user", content=[img])
    parts = m.to_openai()["content"]
    assert len(parts) == 1
    assert parts[0]["type"] == "image_url"
    assert parts[0]["image_url"]["url"] == f"data:image/png;base64,{RED_PIXEL_PNG_B64}"
    assert parts[0]["image_url"]["detail"] == "high"


def test_to_openai_image_without_url_or_data_raises():
    m = Message(role="user", content=[ImageContent()])
    with pytest.raises(ValueError, match="ImageContent requires either"):
        m.to_openai()


def test_to_openai_raw_provider_dict_passes_through():
    raw = {"type": "image_url", "image_url": {"url": "x"}}
    m = Message(role="user", content=[raw])
    assert m.to_openai()["content"] == [raw]


# ---------------------------------------------------------------------------
# ChatAnthropic._format_messages — Anthropic content block shape
# ---------------------------------------------------------------------------


def test_anthropic_format_text_plus_image_url():
    llm = ChatAnthropic(api_key="test")
    m = Message(
        role="user",
        content=[
            TextContent(content="describe"),
            ImageContent.from_url("https://example.com/x.jpg"),
        ],
    )
    formatted = llm._format_messages([m])
    assert formatted[0]["role"] == "user"
    assert formatted[0]["content"] == [
        {"type": "text", "text": "describe"},
        {"type": "image", "source": {"type": "url", "url": "https://example.com/x.jpg"}},
    ]


def test_anthropic_format_image_base64_source():
    llm = ChatAnthropic(api_key="test")
    img = ImageContent.from_bytes(RED_PIXEL_PNG, media_type="image/png")
    m = Message(role="user", content=[img])
    formatted = llm._format_messages([m])
    assert formatted[0]["content"] == [
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": RED_PIXEL_PNG_B64},
        }
    ]


def test_anthropic_format_image_missing_data_raises():
    llm = ChatAnthropic(api_key="test")
    m = Message(role="user", content=[ImageContent()])
    with pytest.raises(ValueError, match="ImageContent requires either"):
        llm._format_messages([m])


def test_anthropic_format_raw_provider_dict_passes_through():
    llm = ChatAnthropic(api_key="test")
    raw = {"type": "image", "source": {"type": "url", "url": "x"}}
    m = Message(role="user", content=[raw])
    assert llm._format_messages([m])[0]["content"] == [raw]


# ---------------------------------------------------------------------------
# JSON roundtrip
# ---------------------------------------------------------------------------


def test_json_roundtrip_rehydrates_content_parts():
    m = Message(
        role="user",
        content=[
            TextContent(content="hello"),
            ImageContent.from_url("https://example.com/x.jpg"),
        ],
    )
    payload = json.dumps(dataclasses.asdict(m))
    m2 = Message(**json.loads(payload))
    assert isinstance(m2.content[0], TextContent)
    assert isinstance(m2.content[1], ImageContent)
    assert m2.content[1].url == "https://example.com/x.jpg"


def test_json_roundtrip_preserves_image_metadata():
    img = ImageContent.from_bytes(b"abc", media_type="image/png", detail="high")
    m = Message(role="user", content=[img])
    m2 = Message(**json.loads(json.dumps(dataclasses.asdict(m))))
    img2 = m2.content[0]
    assert isinstance(img2, ImageContent)
    assert img2.data == base64.b64encode(b"abc").decode("ascii")
    assert img2.media_type == "image/png"
    assert img2.detail == "high"


# ---------------------------------------------------------------------------
# End-to-end: invoke with image goes through the mocked OpenAI client
# ---------------------------------------------------------------------------


def _fake_openai_response(text: str = "red"):
    msg = SimpleNamespace(role="assistant", content=text, tool_calls=None, reasoning_content=None)
    choice = SimpleNamespace(message=msg, finish_reason="stop")
    usage = SimpleNamespace(
        prompt_tokens=10, completion_tokens=1, total_tokens=11, completion_tokens_details=None
    )
    return SimpleNamespace(choices=[choice], usage=usage)


def test_chat_openai_invoke_sends_image_in_wire_format():
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = _fake_openai_response("red")

    llm = ChatOpenAI(api_key="test", model="gpt-4o-mini")
    llm.client = fake_client

    msg = Message(
        role="user",
        content=[
            TextContent(content="What color?"),
            ImageContent.from_bytes(RED_PIXEL_PNG, media_type="image/png"),
        ],
    )
    response = llm.invoke([msg])

    assert response.text == "red"

    # Inspect what the OpenAI client actually received
    _, kwargs = fake_client.chat.completions.create.call_args
    sent_messages = kwargs["messages"]
    assert len(sent_messages) == 1
    sent_content = sent_messages[0]["content"]
    assert sent_content[0] == {"type": "text", "text": "What color?"}
    assert sent_content[1]["type"] == "image_url"
    assert sent_content[1]["image_url"]["url"] == f"data:image/png;base64,{RED_PIXEL_PNG_B64}"
