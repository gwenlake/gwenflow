import json
from time import time

from pydantic import BaseModel, Field


class ResponseOutputItem(BaseModel):
    id: str
    """The id of the output."""

    name: str
    """The name of output."""

    data: list = Field(default_factory=list)
    """A list of data."""

    created_at: int = Field(default_factory=lambda: int(time()))

    def to_dict(self) -> list:
        """Convert the output into a list of dict."""
        return list(self.output)  # type: ignore

    def to_json(self, max_results: int = None) -> str:
        if max_results:
            return json.dumps(self.data[:max_results])
        return json.dumps(self.data)
