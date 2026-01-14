from typing import Any

from pydantic import BaseModel


class ToolCall(BaseModel):
    id: str
    """Unique identifier for tool call."""

    function: str
    """Function called."""

    arguments: dict[str, Any]
    """Arguments to function."""
