from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union


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
    extra: Optional[dict] = None

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
