import json
from typing import Optional, Any, Dict
from typing_extensions import Literal
from pydantic import BaseModel, Field
from time import time

from gwenflow.types.message import Message
from gwenflow.types.usage import Usage

  
class Function(BaseModel):
    name: str
    arguments: str

class ToolCall(BaseModel):
    id: str
    function: Function
    type: Literal["function"]
    
class ToolResponse(BaseModel):
    tool_call_id: str
    tool_name: str
    tool_args: Optional[Dict[str, Any]] = None
    tool_call_error: Optional[bool] = None
    result: Optional[str] = None
    usage: Optional[Usage] = None
    created_at: int = Field(default_factory=lambda: int(time()))

    def to_message(self) -> Message:
        return Message(
            role="tool",
            tool_call_id=self.tool_call_id,
            tool_name=self.tool_name,
            content=self.result,
        )
