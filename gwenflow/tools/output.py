from pydantic import BaseModel, Field
from time import time
import json

from gwenflow.types.message import Message


class ToolOutput(BaseModel):

    id: str
    """The id of the output."""

    name: str
    """The name of output (name of the tool used to generate this output."""

    output: list = Field(default_factory=list)
    """A list of output data."""

    created_at: int = Field(default_factory=lambda: int(time()))

    def to_dict(self) -> list:
        """Convert the output into a list of dict."""
        return [d for d in self.output]  # type: ignore

    def to_json_str(self, max_result: int) -> str:
        return json.dumps(self.output[:max_result])

    def to_message(self, max_result: int = 20) -> Message:
        return Message(
            role="tool",
            tool_call_id=self.id,
            tool_name=self.name,
            content=self.to_json_str(max_result=max_result),
        )