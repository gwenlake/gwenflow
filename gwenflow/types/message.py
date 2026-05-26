import base64
import json
import mimetypes
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Annotated, Any

from pydantic import Discriminator
from typing_extensions import Literal


@dataclass(kw_only=True)
class TextContent:
    """A plain-text part of a multi-modal content list.

    Used when `Message.content` is a list rather than a bare string — typically
    alongside `ImageContent`/`AudioContent`/`FileContent`. When `Message.content`
    is just text, providers send it as a string and this wrapper isn't needed.
    """

    content: str
    kind: Literal["text"] = "text"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TextContent":
        return TextContent(**data)


@dataclass(kw_only=True)
class ThinkingContent:
    """The model's internal reasoning, exposed as a separate part.

    Populated from `reasoning_content` (DeepSeek, Mistral, Gemma/Qwen via
    --reasoning-parser), `reasoning` (OpenAI gpt-5/o-series), or thinking
    blocks (Anthropic extended thinking). `extra` carries provider-specific
    metadata that must survive a roundtrip — notably the Anthropic block
    signature, which is required to echo thinking back on tool-use turns.
    """

    content: str
    extra: dict[str, Any] = field(default_factory=dict)
    kind: Literal["thinking"] = "thinking"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ThinkingContent":
        return ThinkingContent(**data)


@dataclass(kw_only=True)
class ImageContent:
    """An image part for multi-modal input (and, eventually, output).

    Use exactly one of `url` or `data`. `data` is base64-encoded bytes; combined
    with `media_type` it becomes a `data:` URI on the wire for OpenAI-family
    providers, or an explicit base64 source for Anthropic.
    """

    url: str | None = None
    data: str | None = None
    media_type: str = "image/jpeg"
    detail: Literal["auto", "low", "high"] | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    kind: Literal["image"] = "image"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ImageContent":
        return ImageContent(**data)

    @classmethod
    def from_url(cls, url: str, detail: str | None = None) -> "ImageContent":
        return cls(url=url, detail=detail)

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        media_type: str = "image/jpeg",
        detail: str | None = None,
    ) -> "ImageContent":
        encoded = base64.b64encode(data).decode("ascii")
        return cls(data=encoded, media_type=media_type, detail=detail)

    @classmethod
    def from_path(cls, path: str | Path, detail: str | None = None) -> "ImageContent":
        p = Path(path)
        guessed, _ = mimetypes.guess_type(str(p))
        media_type = guessed or "image/jpeg"
        return cls.from_bytes(p.read_bytes(), media_type=media_type, detail=detail)


@dataclass(kw_only=True)
class AudioContent:
    """An audio part for multi-modal input (and output, for audio-capable models).

    `data` is base64-encoded audio bytes; `format` describes the container.
    Supported by OpenAI gpt-4o-audio models and Google Gemini. Not supported by
    Anthropic — providers without audio will raise on translation.
    """

    data: str
    format: Literal["wav", "mp3", "flac", "opus", "pcm16"] = "wav"
    transcript: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    kind: Literal["audio"] = "audio"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AudioContent":
        return AudioContent(**data)

    @classmethod
    def from_bytes(cls, data: bytes, format: str = "wav") -> "AudioContent":
        return cls(data=base64.b64encode(data).decode("ascii"), format=format)

    @classmethod
    def from_path(cls, path: str | Path) -> "AudioContent":
        p = Path(path)
        ext = p.suffix.lstrip(".").lower() or "wav"
        return cls.from_bytes(p.read_bytes(), format=ext)


@dataclass(kw_only=True)
class FileContent:
    """A file part for multi-modal input — typically a PDF.

    Provide exactly one of:
      - `file_id`: ID returned by the provider's Files API after upload.
      - `data`: base64-encoded bytes, sent inline.
      - `url`: hosted URL (Anthropic accepts this for PDFs).
    """

    file_id: str | None = None
    data: str | None = None
    url: str | None = None
    filename: str | None = None
    media_type: str = "application/pdf"
    extra: dict[str, Any] = field(default_factory=dict)
    kind: Literal["file"] = "file"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FileContent":
        return FileContent(**data)

    @classmethod
    def from_file_id(cls, file_id: str) -> "FileContent":
        return cls(file_id=file_id)

    @classmethod
    def from_url(cls, url: str, media_type: str = "application/pdf") -> "FileContent":
        return cls(url=url, media_type=media_type)

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        media_type: str = "application/pdf",
        filename: str | None = None,
    ) -> "FileContent":
        return cls(
            data=base64.b64encode(data).decode("ascii"),
            media_type=media_type,
            filename=filename,
        )

    @classmethod
    def from_path(cls, path: str | Path) -> "FileContent":
        p = Path(path)
        guessed, _ = mimetypes.guess_type(str(p))
        media_type = guessed or "application/pdf"
        return cls.from_bytes(p.read_bytes(), media_type=media_type, filename=p.name)


@dataclass(kw_only=True)
class ToolCall:
    """A tool invocation the model wants the runtime to execute.

    `arguments` is the raw JSON string produced by the model (kept as a string
    so partial streaming deltas can be concatenated), or a parsed dict once
    finalized. `to_message_dict()` produces the OpenAI nested `function`-call
    shape used inside an assistant Message's `tool_calls` list.
    """

    id: str | None = None
    name: str
    arguments: str | dict[str, Any] | None = None
    kind: Literal["tool-call"] = "tool-call"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolCall":
        return ToolCall(**data)

    def to_message_dict(self) -> dict[str, Any]:
        args = self.arguments if isinstance(self.arguments, str) else json.dumps(self.arguments)
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": args,
            },
        }


MessageContent = Annotated[TextContent | ImageContent | AudioContent | FileContent | ToolCall, Discriminator("kind")]
"""Parts that can appear in a Message's visible content (no thinking — that lives in Message.thinking_parts)."""

ResponsePart = Annotated[TextContent | ThinkingContent | ImageContent | AudioContent | ToolCall, Discriminator("kind")]
"""Parts that can appear in a ModelResponse — includes thinking, since it's part of the raw model output."""


_PART_BY_KIND = {
    "text": TextContent,
    "image": ImageContent,
    "audio": AudioContent,
    "file": FileContent,
    "thinking": ThinkingContent,
    "tool-call": ToolCall,
}


def _rehydrate_part(part: Any) -> Any:
    """Convert a dict with a `kind` discriminator back into its typed dataclass.

    Pass-through for already-typed objects and for raw provider-shaped dicts
    (those without a `kind` field — e.g. {"type": "image_url", ...}).
    """
    if not isinstance(part, dict):
        return part
    kind = part.get("kind")
    cls = _PART_BY_KIND.get(kind) if kind else None
    if cls is None:
        return part
    return cls(**part)


SYSTEM = "system"
USER = "user"
ASSISTANT = "assistant"
TOOL = "tool"

_VALID_ROLES = {USER, ASSISTANT, SYSTEM, TOOL}


@dataclass
class Message:
    """One turn in a conversation, in gwenflow's provider-neutral form.

    `role` is system/user/assistant/tool. `content` is a string for plain text
    or a list of `MessageContent` parts for multi-modal input (text + image +
    audio + file). Each provider's adapter translates this to its own wire
    shape — `to_openai()` for the OpenAI-compatible family, and
    `ChatAnthropic._format_messages` for Anthropic's content-block API.
    """

    role: str
    content: str | list[MessageContent] | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    thinking_parts: list[ThinkingContent] | None = None
    """Thinking parts to echo back on the next turn.
    Only Anthropic extended-thinking needs these — signatures (stored in
    ThinkingContent.extra) must be replayed verbatim with tool use.
    Other providers ignore the field."""
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.role not in _VALID_ROLES:
            raise ValueError(f"{self.role} must be one of {','.join(_VALID_ROLES)}")
        # Normalize after a to_dict/from_dict roundtrip: asdict produces plain
        # dicts; rehydrate them into their dataclass form.
        if self.thinking_parts is not None:
            self.thinking_parts = [
                ThinkingContent.from_dict(p) if isinstance(p, dict) else p for p in self.thinking_parts
            ]
        if isinstance(self.content, list):
            self.content = [_rehydrate_part(p) for p in self.content]

    def __repr__(self):
        return f"Message({self.to_dict()})"

    def to_dict(self, **kwargs: Any) -> dict[str, Any]:
        message_dict = {
            "role": self.role,
            "content": self.content,
            "name": self.name,
            "tool_call_id": self.tool_call_id,
            "tool_calls": self.tool_calls,
            "thinking_parts": self.thinking_parts,
            "extra": self.extra,
        }
        return {
            k: v for k, v in message_dict.items() if v is not None and not (isinstance(v, (list, dict)) and len(v) == 0)
        }

    def to_openai(self) -> dict[str, Any]:
        content: Any = self.content
        if isinstance(content, list):
            content = [_part_to_openai(p) for p in content]

        message_dict: dict[str, Any] = {
            "role": self.role,
            "content": content,
            "name": self.name,
            "tool_call_id": self.tool_call_id,
            "tool_calls": self.tool_calls,
        }
        message_dict = {k: v for k, v in message_dict.items() if v is not None}

        if self.tool_calls is not None and len(self.tool_calls) == 0:
            message_dict["tool_calls"] = None

        return message_dict


def _part_to_openai(part: Any) -> dict[str, Any]:
    """Translate a MessageContent part to the OpenAI Chat-Completions wire shape."""
    if isinstance(part, dict):
        return part  # already wire-shaped, pass through
    if isinstance(part, TextContent):
        return {"type": "text", "text": part.content}
    if isinstance(part, ImageContent):
        if part.url:
            url = part.url
        elif part.data:
            url = f"data:{part.media_type};base64,{part.data}"
        else:
            raise ValueError("ImageContent requires either `url` or `data`.")
        image_url: dict[str, Any] = {"url": url}
        if part.detail:
            image_url["detail"] = part.detail
        return {"type": "image_url", "image_url": image_url}
    if isinstance(part, AudioContent):
        return {"type": "input_audio", "input_audio": {"data": part.data, "format": part.format}}
    if isinstance(part, FileContent):
        file: dict[str, Any] = {}
        if part.file_id:
            file["file_id"] = part.file_id
        elif part.data:
            file["file_data"] = f"data:{part.media_type};base64,{part.data}"
            if part.filename:
                file["filename"] = part.filename
        elif part.url:
            raise ValueError(
                "OpenAI Chat Completions does not accept URL-only files; upload via Files API "
                "and use FileContent.from_file_id(...) instead."
            )
        else:
            raise ValueError("FileContent requires `file_id`, `data`, or `url`.")
        return {"type": "file", "file": file}
    raise TypeError(f"Unsupported content part for OpenAI wire format: {type(part).__name__}")
