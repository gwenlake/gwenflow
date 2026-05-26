"""Unit tests for audio and file (PDF) input (Phase 2 multi-modal).

Covers:
- AudioContent / FileContent constructors and helpers
- Message.to_openai() translation for audio (input_audio) and file (file / file_id)
- ChatAnthropic._format_messages translation for documents, and clean rejection of audio
- JSON roundtrip rehydration
"""

import base64
import dataclasses
import json
from pathlib import Path

import pytest

from gwenflow.llms.anthropic import ChatAnthropic
from gwenflow.types import AudioContent, FileContent, Message, TextContent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FAKE_WAV = b"RIFFfakefakefake"
FAKE_WAV_B64 = base64.b64encode(FAKE_WAV).decode("ascii")

FAKE_PDF = b"%PDF-1.4\n%fakecontent\n%%EOF"
FAKE_PDF_B64 = base64.b64encode(FAKE_PDF).decode("ascii")


@pytest.fixture
def wav_path(tmp_path: Path) -> Path:
    p = tmp_path / "clip.wav"
    p.write_bytes(FAKE_WAV)
    return p


@pytest.fixture
def pdf_path(tmp_path: Path) -> Path:
    p = tmp_path / "report.pdf"
    p.write_bytes(FAKE_PDF)
    return p


# ---------------------------------------------------------------------------
# AudioContent constructors
# ---------------------------------------------------------------------------


def test_audio_from_bytes_default_wav():
    a = AudioContent.from_bytes(FAKE_WAV)
    assert a.data == FAKE_WAV_B64
    assert a.format == "wav"


def test_audio_from_bytes_mp3():
    a = AudioContent.from_bytes(FAKE_WAV, format="mp3")
    assert a.format == "mp3"


def test_audio_from_path_infers_format(wav_path: Path):
    a = AudioContent.from_path(wav_path)
    assert a.format == "wav"
    assert a.data == FAKE_WAV_B64


def test_audio_from_path_no_extension(tmp_path: Path):
    p = tmp_path / "noext"
    p.write_bytes(b"x")
    a = AudioContent.from_path(p)
    assert a.format == "wav"  # fallback default


# ---------------------------------------------------------------------------
# FileContent constructors
# ---------------------------------------------------------------------------


def test_file_from_bytes():
    f = FileContent.from_bytes(FAKE_PDF, filename="report.pdf")
    assert f.data == FAKE_PDF_B64
    assert f.media_type == "application/pdf"
    assert f.filename == "report.pdf"


def test_file_from_path_infers_media_type(pdf_path: Path):
    f = FileContent.from_path(pdf_path)
    assert f.media_type == "application/pdf"
    assert f.filename == "report.pdf"


def test_file_from_file_id():
    f = FileContent.from_file_id("file-abc123")
    assert f.file_id == "file-abc123"
    assert f.data is None and f.url is None


def test_file_from_url():
    f = FileContent.from_url("https://example.com/doc.pdf")
    assert f.url == "https://example.com/doc.pdf"
    assert f.media_type == "application/pdf"


# ---------------------------------------------------------------------------
# OpenAI wire shape
# ---------------------------------------------------------------------------


def test_to_openai_audio_input_audio():
    a = AudioContent.from_bytes(FAKE_WAV, format="wav")
    m = Message(role="user", content=[TextContent(content="transcribe"), a])
    parts = m.to_openai()["content"]
    assert parts[0] == {"type": "text", "text": "transcribe"}
    assert parts[1] == {"type": "input_audio", "input_audio": {"data": FAKE_WAV_B64, "format": "wav"}}


def test_to_openai_file_inline_base64():
    f = FileContent.from_bytes(FAKE_PDF, filename="report.pdf")
    m = Message(role="user", content=[f])
    parts = m.to_openai()["content"]
    assert parts[0] == {
        "type": "file",
        "file": {
            "file_data": f"data:application/pdf;base64,{FAKE_PDF_B64}",
            "filename": "report.pdf",
        },
    }


def test_to_openai_file_inline_no_filename():
    f = FileContent.from_bytes(FAKE_PDF)  # no filename
    m = Message(role="user", content=[f])
    parts = m.to_openai()["content"]
    assert "filename" not in parts[0]["file"]


def test_to_openai_file_by_id():
    f = FileContent.from_file_id("file-abc123")
    m = Message(role="user", content=[f])
    parts = m.to_openai()["content"]
    assert parts[0] == {"type": "file", "file": {"file_id": "file-abc123"}}


def test_to_openai_file_url_only_raises():
    m = Message(role="user", content=[FileContent.from_url("https://x/y.pdf")])
    with pytest.raises(ValueError, match="Files API"):
        m.to_openai()


def test_to_openai_file_empty_raises():
    m = Message(role="user", content=[FileContent()])
    with pytest.raises(ValueError, match="FileContent requires"):
        m.to_openai()


# ---------------------------------------------------------------------------
# Anthropic wire shape
# ---------------------------------------------------------------------------


def test_anthropic_rejects_audio_input():
    llm = ChatAnthropic(api_key="test")
    a = AudioContent.from_bytes(FAKE_WAV)
    m = Message(role="user", content=[a])
    with pytest.raises(NotImplementedError, match="audio"):
        llm._format_messages([m])


def test_anthropic_document_base64_source():
    llm = ChatAnthropic(api_key="test")
    f = FileContent.from_bytes(FAKE_PDF)
    m = Message(role="user", content=[f])
    formatted = llm._format_messages([m])
    assert formatted[0]["content"] == [
        {
            "type": "document",
            "source": {"type": "base64", "media_type": "application/pdf", "data": FAKE_PDF_B64},
        }
    ]


def test_anthropic_document_url_source():
    llm = ChatAnthropic(api_key="test")
    f = FileContent.from_url("https://example.com/r.pdf")
    m = Message(role="user", content=[f])
    formatted = llm._format_messages([m])
    assert formatted[0]["content"] == [
        {"type": "document", "source": {"type": "url", "url": "https://example.com/r.pdf"}}
    ]


def test_anthropic_rejects_files_api_id():
    llm = ChatAnthropic(api_key="test")
    f = FileContent.from_file_id("file-abc")
    m = Message(role="user", content=[f])
    with pytest.raises(ValueError, match="file_id"):
        llm._format_messages([m])


def test_anthropic_empty_file_raises():
    llm = ChatAnthropic(api_key="test")
    m = Message(role="user", content=[FileContent()])
    with pytest.raises(ValueError, match="FileContent requires"):
        llm._format_messages([m])


# ---------------------------------------------------------------------------
# JSON roundtrip
# ---------------------------------------------------------------------------


def test_json_roundtrip_audio():
    m = Message(role="user", content=[AudioContent.from_bytes(FAKE_WAV, format="mp3")])
    m2 = Message(**json.loads(json.dumps(dataclasses.asdict(m))))
    audio = m2.content[0]
    assert isinstance(audio, AudioContent)
    assert audio.format == "mp3"
    assert audio.data == FAKE_WAV_B64


def test_json_roundtrip_file_preserves_all_fields():
    m = Message(
        role="user",
        content=[
            FileContent.from_bytes(FAKE_PDF, filename="r.pdf"),
            FileContent.from_file_id("file-x"),
            FileContent.from_url("https://x/y.pdf"),
        ],
    )
    m2 = Message(**json.loads(json.dumps(dataclasses.asdict(m))))
    assert isinstance(m2.content[0], FileContent)
    assert m2.content[0].filename == "r.pdf"
    assert m2.content[1].file_id == "file-x"
    assert m2.content[2].url == "https://x/y.pdf"


def test_json_roundtrip_mixed_parts():
    m = Message(
        role="user",
        content=[
            TextContent(content="hello"),
            AudioContent.from_bytes(FAKE_WAV),
            FileContent.from_bytes(FAKE_PDF),
        ],
    )
    m2 = Message(**json.loads(json.dumps(dataclasses.asdict(m))))
    assert isinstance(m2.content[0], TextContent)
    assert isinstance(m2.content[1], AudioContent)
    assert isinstance(m2.content[2], FileContent)
