
from typing import List, Optional, Any
from pydantic import BaseModel, Field
from time import time


class AgentResponse(BaseModel):
    content: Optional[Any] = None
    content_type: str = "str"
    delta: Optional[str] = None
    messages: Optional[List] = None
    agent: Optional[Any] = None
    tools: Optional[List[Any]] = None
    created_at: int = Field(default_factory=lambda: int(time()))
