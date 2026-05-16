import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from gwenflow.types.message import Message, MessageContent, TextContent, ThinkingContent, ToolCall
from gwenflow.types.usage import AgentUsage, RequestUsage
from gwenflow.utils.utils import now_utc


@dataclass
class ModelResponse:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parts: Sequence[MessageContent] = field(default_factory=list)
    parsed: Any | None = None
    finish_reason: str | None = None
    usage: RequestUsage = field(default_factory=RequestUsage)
    created_at: datetime = field(default_factory=now_utc)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_message(self) -> Message:
        return Message(**self.to_dict())

    def to_openai(self) -> Dict[str, Any]:
        """Return this response as an OpenAI ChatCompletion-shaped dict."""
        message: Dict[str, Any] = {"role": "assistant", "content": self.text}

        if self.thinking is not None:
            message["reasoning_content"] = self.thinking

        if self.tool_calls:
            message["tool_calls"] = [tc.to_message_dict() for tc in self.tool_calls]

        completion: Dict[str, Any] = {
            "id": self.id,
            "object": "chat.completion",
            "created": int(self.created_at.timestamp()),
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": self.finish_reason,
                }
            ],
        }

        if self.usage is not None:
            usage_dict: Dict[str, Any] = {
                "prompt_tokens": self.usage.input_tokens,
                "completion_tokens": self.usage.output_tokens,
                "total_tokens": self.usage.total_tokens,
            }
            if self.usage.details:
                usage_dict["completion_tokens_details"] = self.usage.details
            completion["usage"] = usage_dict

        return completion

    @property
    def text(self) -> str | None:
        """Get the text in the response."""
        texts: list[str] = []
        last_part: MessageContent | None = None
        for part in self.parts:
            if isinstance(part, TextContent):
                if isinstance(last_part, TextContent):
                    texts[-1] += part.content
                else:
                    texts.append(part.content)
            last_part = part
        if not texts:
            return None

        return "\n\n".join(texts)

    @property
    def content(self) -> str | None:
        return self.text

    @property
    def thinking(self) -> str | None:
        """Get the thinking in the response."""
        thinking_parts = [part.content for part in self.parts if isinstance(part, ThinkingContent)]
        if not thinking_parts:
            return None
        return "\n\n".join(thinking_parts)

    @property
    def tool_calls(self) -> List[ToolCall]:
        return [part for part in self.parts if isinstance(part, ToolCall)]


@dataclass
class AgentResponse:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str | None = None
    parsed: Any | None = None
    reasoning_content: str | None = None
    messages: list[Message] = field(default_factory=list)
    finish_reason: str | None = None
    usage: AgentUsage = field(default_factory=AgentUsage)
    created_at: datetime = field(default_factory=now_utc)


@dataclass
class ToolResponse:
    tool_call_id: str
    tool_name: str
    tool_args: Optional[Dict[str, Any]] = None
    tool_call_error: Optional[bool] = None
    content: str | None = None
    created_at: datetime = field(default_factory=now_utc)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_message(self) -> Message:
        return Message(
            role="tool",
            tool_call_id=self.tool_call_id,
            content=self.content,
        )
