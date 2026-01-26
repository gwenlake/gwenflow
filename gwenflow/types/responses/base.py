from typing import List, Literal, Optional

from pydantic import BaseModel


class ResponseBase(BaseModel):
    sequence_number: int

class Logprob(BaseModel): # TODO move this later 
    token: str
    logprob: float
    bytes: Optional[List[int]] = None

class ReasoningItem(BaseModel): # TODO move this later
    effort: Literal['low', 'medium', 'high']
    summary: Literal['auto', 'concise', 'detailed']
