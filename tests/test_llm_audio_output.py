"""Unit tests for audio output handling (Phase 3 multi-modal).

Covers:
- ChatOpenAI._parse_response: turns a Chat-Completions audio response
  (`message.audio`) into an AudioContent part on ModelResponse.
- _audio_delta_to_part: builds AudioContent chunks from streaming deltas.
- ModelResponse.audio: aggregates streamed chunks into one AudioContent.
- ModelResponse.images: convenience accessor for image parts.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gwenflow.llms.openai import ChatOpenAI, _audio_delta_to_part
from gwenflow.types import AudioContent, ImageContent, ModelResponse, TextContent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _audio_completion(
    *,
    data: str = "BASE64AUDIODATA",
    transcript: str | None = "Hello world.",
    audio_id: str | None = "audio_abc",
    expires_at: int | None = 1234567890,
    audio_format: str | None = "wav",
):
    audio = SimpleNamespace(
        id=audio_id,
        data=data,
        expires_at=expires_at,
        transcript=transcript,
        format=audio_format,
    )
    msg = SimpleNamespace(
        role="assistant",
        content=None,
        tool_calls=None,
        reasoning_content=None,
        reasoning=None,
        audio=audio,
    )
    choice = SimpleNamespace(message=msg, finish_reason="stop")
    usage = SimpleNamespace(
        prompt_tokens=5, completion_tokens=10, total_tokens=15, completion_tokens_details=None
    )
    return SimpleNamespace(choices=[choice], usage=usage)


def _audio_chunk(data: str | None = None, transcript: str | None = None, id: str | None = None, expires_at: int | None = None):
    """Streaming-shaped audio delta."""
    return SimpleNamespace(data=data, transcript=transcript, id=id, expires_at=expires_at)


# ---------------------------------------------------------------------------
# _parse_response — non-streaming audio output
# ---------------------------------------------------------------------------


def test_parse_response_extracts_audio_output():
    llm = ChatOpenAI(api_key="test", model="gpt-4o-audio-preview")
    resp = llm._parse_response(_audio_completion())

    audio = resp.audio
    assert isinstance(audio, AudioContent)
    assert audio.data == "BASE64AUDIODATA"
    assert audio.transcript == "Hello world."
    assert audio.format == "wav"
    assert audio.extra == {"id": "audio_abc", "expires_at": 1234567890}


def test_parse_response_audio_without_optional_fields():
    llm = ChatOpenAI(api_key="test", model="gpt-4o-audio-preview")
    resp = llm._parse_response(_audio_completion(transcript=None, expires_at=None, audio_id=None))

    audio = resp.audio
    assert audio.data == "BASE64AUDIODATA"
    assert audio.transcript is None
    assert audio.extra == {}


def test_parse_response_skips_audio_with_no_data():
    """If `audio` exists but has no data, no AudioContent part is added."""
    audio = SimpleNamespace(id="x", data=None, expires_at=None, transcript=None, format="wav")
    msg = SimpleNamespace(
        role="assistant", content="text only", tool_calls=None, reasoning_content=None, reasoning=None, audio=audio
    )
    choice = SimpleNamespace(message=msg, finish_reason="stop")
    usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2, completion_tokens_details=None)
    completion = SimpleNamespace(choices=[choice], usage=usage)

    llm = ChatOpenAI(api_key="test")
    resp = llm._parse_response(completion)
    assert resp.audio is None
    assert resp.text == "text only"


def test_parse_response_audio_does_not_have_format_attribute_uses_default():
    """When the SDK chunk omits `format` (older SDKs), default to 'wav'."""
    audio = SimpleNamespace(id="x", data="DATA", expires_at=1, transcript=None)
    # Note: no .format attribute
    msg = SimpleNamespace(
        role="assistant", content=None, tool_calls=None, reasoning_content=None, reasoning=None, audio=audio
    )
    choice = SimpleNamespace(message=msg, finish_reason="stop")
    usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2, completion_tokens_details=None)
    completion = SimpleNamespace(choices=[choice], usage=usage)

    llm = ChatOpenAI(api_key="test")
    resp = llm._parse_response(completion)
    assert resp.audio.format == "wav"


# ---------------------------------------------------------------------------
# _audio_delta_to_part — streaming
# ---------------------------------------------------------------------------


def test_audio_delta_none_returns_none():
    assert _audio_delta_to_part(None) is None


def test_audio_delta_empty_payload_returns_none():
    assert _audio_delta_to_part(_audio_chunk(data=None, transcript=None)) is None


def test_audio_delta_with_data():
    part = _audio_delta_to_part(_audio_chunk(data="CHUNK1", id="a1", expires_at=99))
    assert part.data == "CHUNK1"
    assert part.transcript is None
    assert part.extra == {"id": "a1", "expires_at": 99}


def test_audio_delta_with_transcript_only():
    part = _audio_delta_to_part(_audio_chunk(transcript="Hi "))
    assert part.data == ""
    assert part.transcript == "Hi "


def test_audio_delta_with_both():
    part = _audio_delta_to_part(_audio_chunk(data="X", transcript="Y", id="i"))
    assert part.data == "X"
    assert part.transcript == "Y"
    assert part.extra == {"id": "i"}


# ---------------------------------------------------------------------------
# ModelResponse.audio aggregation
# ---------------------------------------------------------------------------


def test_model_response_audio_none_when_no_audio_parts():
    mr = ModelResponse(parts=[TextContent(content="hi")])
    assert mr.audio is None


def test_model_response_audio_concatenates_chunks():
    mr = ModelResponse(parts=[
        TextContent(content="unrelated"),
        AudioContent(data="AAA", transcript="Hi ", extra={"id": "x"}),
        AudioContent(data="BBB", transcript="there.", extra={"expires_at": 1000}),
    ])
    agg = mr.audio
    assert agg.data == "AAABBB"
    assert agg.transcript == "Hi there."
    assert agg.extra == {"id": "x", "expires_at": 1000}


def test_model_response_audio_format_from_last_chunk():
    mr = ModelResponse(parts=[
        AudioContent(data="A", format="wav"),
        AudioContent(data="B", format="mp3"),
    ])
    assert mr.audio.format == "mp3"


def test_model_response_audio_empty_chunks_skipped():
    """Chunks with empty data but a transcript should still be included."""
    mr = ModelResponse(parts=[
        AudioContent(data="", transcript="Hello"),
        AudioContent(data="DATA", transcript=" world"),
    ])
    assert mr.audio.data == "DATA"
    assert mr.audio.transcript == "Hello world"


# ---------------------------------------------------------------------------
# ModelResponse.images
# ---------------------------------------------------------------------------


def test_model_response_images_empty_by_default():
    assert ModelResponse(parts=[TextContent(content="hi")]).images == []


def test_model_response_images_returns_only_image_parts():
    mr = ModelResponse(parts=[
        ImageContent(url="x"),
        TextContent(content="y"),
        ImageContent(data="Z", media_type="image/png"),
    ])
    assert len(mr.images) == 2
    assert all(isinstance(p, ImageContent) for p in mr.images)
    assert mr.images[0].url == "x"
    assert mr.images[1].media_type == "image/png"


# ---------------------------------------------------------------------------
# End-to-end mocked invoke
# ---------------------------------------------------------------------------


def test_chat_openai_invoke_returns_audio_response():
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = _audio_completion(
        data="OUTPUTAUDIO", transcript="The answer is 42."
    )

    llm = ChatOpenAI(api_key="test", model="gpt-4o-audio-preview")
    llm.client = fake_client

    response = llm.invoke("What's the meaning of life?")

    assert response.audio is not None
    assert response.audio.data == "OUTPUTAUDIO"
    assert response.audio.transcript == "The answer is 42."
