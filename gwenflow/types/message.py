
from pydantic import BaseModel, ConfigDict
from typing import Any, Dict, Optional, Union, List


class ChatMessage(BaseModel):
    """Chat message class."""

    role: str
    """The role of the messages author (system, user, assistant or tool)."""

    content: Union[str, list[Union[str, dict]]]
    """Content of the message."""

    name: Optional[str] = None
    """An optional name for the participant."""

    tool_call_id: Optional[str] = None
    """Tool call that this message is responding to."""
    
    tool_calls: Optional[List[Dict[str, Any]]] = None
    """The tool calls generated by the model, such as function calls."""

    model_config = ConfigDict(extra="allow", populate_by_name=True, arbitrary_types_allowed=True)


    def __str__(self) -> str:
        return f"{self.role}: {self.content}"

    def to_dict(self, **kwargs: Any) -> Dict[str, Any]:
        return self.model_dump(**kwargs)

