import uuid
from datetime import datetime
from typing import Any, List, Optional, Annotated, Dict
from collections.abc import Sequence
from dataclasses import dataclass, field, asdict

from pydantic import Discriminator
from typing_extensions import Literal

from gwenflow.types.message import Message
from gwenflow.types.usage import RequestUsage, AgentUsage
from gwenflow.utils.utils import now_utc


@dataclass(kw_only=True)
class TextPart:
    id: str | None = None
    content: str
    part_kind: Literal['text'] = 'text'

@dataclass(kw_only=True)
class ThinkingPart:
    id: str | None = None
    content: str
    part_kind: Literal['thinking'] = 'thinking'

@dataclass(kw_only=True)
class ToolCallPart:
    id: str | None = None
    function: str | None = None
    arguments: str | dict[str, Any] | None = None
    part_kind: Literal['tool-call'] = 'tool-call'

    def to_openai(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.function,
                "arguments": self.arguments,
            }
        }

ModelResponsePart = Annotated[TextPart | ToolCallPart | ThinkingPart, Discriminator('part_kind')]

@dataclass
class ModelResponse:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parts: Sequence[ModelResponsePart] = field(default_factory=list)
    parsed: Any | None = None
    finish_reason: str | None = None
    usage: RequestUsage = field(default_factory=RequestUsage)
    created_at: datetime = field(default_factory=now_utc)

    def to_message(self) -> Message:
        return Message(**self.model_dump(exclude_unset=True))

    @property
    def text(self) -> str | None:
        """Get the text in the response."""
        texts: list[str] = []
        last_part: ModelResponsePart | None = None
        for part in self.parts:
            if isinstance(part, TextPart):
                if isinstance(last_part, TextPart):
                    texts[-1] += part.content
                else:
                    texts.append(part.content)
            last_part = part
        if not texts:
            return None

        return '\n\n'.join(texts)

    @property
    def content(self) -> str | None:
        """Get the text in the response."""
        return self.text

    @property
    def thinking(self) -> str | None:
        """Get the thinking in the response."""
        thinking_parts = [part.content for part in self.parts if isinstance(part, ThinkingPart)]
        if not thinking_parts:
            return None
        return '\n\n'.join(thinking_parts)

    @property
    def tool_calls(self) -> List[ToolCallPart]:
        return [part for part in self.parts if isinstance(part, ToolCallPart)]

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