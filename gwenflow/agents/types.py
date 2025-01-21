
from typing import List, Optional, Any
from pydantic import BaseModel, Field
from time import time

class ReActAgentAction(BaseModel):
    thought: str
    action: str
    action_input: str
    result : Optional[str] = None
    text: str

class ReActAgentFinish(BaseModel):
    thought: str
    final_answer: str
    text: str

class AgentResponse(BaseModel):
    content: Optional[Any] = None
    content_type: str = "str"
    delta: Optional[str] = None
    messages: Optional[List] = None
    agent: Optional[Any] = None
    tools: Optional[List[Any]] = None
    created_at: int = Field(default_factory=lambda: int(time()))
