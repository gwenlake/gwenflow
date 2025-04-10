from pydantic import BaseModel, Field
from time import time
import json


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

    def to_json(self, max_results: int = None) -> str:
        if max_results:
            return json.dumps(self.output[:max_results])
        return json.dumps(self.output)
