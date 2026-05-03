from typing import Any, List, Optional, Annotated, Dict, Union
from typing_extensions import Literal
from dataclasses import dataclass, field, asdict
from pydantic import Discriminator
import json

@dataclass(kw_only=True)
class TextContent:
    content: str
    kind: Literal['text'] = 'text'
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TextContent":
        return TextContent(**data)


@dataclass(kw_only=True)
class ThinkingContent:
    content: str
    extra: dict[str, Any] = field(default_factory=dict)
    kind: Literal['thinking'] = 'thinking'

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ThinkingContent":
        return ThinkingContent(**data)


@dataclass(kw_only=True)
class ToolCall:
    id: str | None = None
    name: str
    arguments: str | dict[str, Any] | None = None
    kind: Literal['tool-call'] = 'tool-call'

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
            }
        }
    
MessageContent = Annotated[TextContent | ThinkingContent | ToolCall, Discriminator('kind')]


SYSTEM = "system"
USER = "user"
ASSISTANT = "assistant"
TOOL = "tool"

_VALID_ROLES = {USER, ASSISTANT, SYSTEM, TOOL}


@dataclass
class Message:
    role: str
    content: Optional[Union[str, List[Union[str, dict]]]] = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    reasoning_content: Optional[str] = None
    extra: dict[str, Any] | None = None

    def __post_init__(self):
        if self.role not in _VALID_ROLES:
            raise ValueError(f"{self.role} must be one of {','.join(_VALID_ROLES)}")

    def __repr__(self):
        return f"Message({self.to_dict()})"

    def to_dict(self, **kwargs: Any) -> Dict[str, Any]:
        message_dict = {
            "role": self.role,
            "content": self.content,
            "name": self.name,
            "tool_call_id": self.tool_call_id,
            "tool_calls": self.tool_calls,
            "reasoning_content": self.reasoning_content,
            "extra": self.extra,
        }
        return {
            k: v for k, v in message_dict.items() if v is not None and not (isinstance(v, (list, dict)) and len(v) == 0)
        }

    def to_openai(self) -> Dict[str, Any]:
        message_dict: Dict[str, Any] = {
            "role": self.role,
            "content": self.content,
            "name": self.name,
            "tool_call_id": self.tool_call_id,
            "tool_calls": self.tool_calls,
        }
        message_dict = {k: v for k, v in message_dict.items() if v is not None}

        if self.tool_calls is not None and len(self.tool_calls) == 0:
            message_dict["tool_calls"] = None

        return message_dict