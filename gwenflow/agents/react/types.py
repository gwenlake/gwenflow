
from typing import Optional
from pydantic import BaseModel


class ActionReasoningStep(BaseModel):
    thought: str
    action: Optional[str] = None
    action_input: Optional[str] = None
    response : Optional[str] = None
    is_done: bool = False
