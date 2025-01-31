
from typing import Optional, Dict
from pydantic import BaseModel


class ActionReasoningStep(BaseModel):
    thought: str
    action: Optional[str] = None
    action_input: Optional[Dict] = None
    response : Optional[str] = None
    is_done: bool = False
