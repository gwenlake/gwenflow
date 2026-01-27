from typing import List, Literal, Optional, Union, Dict, Any

from pydantic import Field, RootModel, model_validator

from gwenflow.types.responses.base import Logprob, ResponseBase
from gwenflow.types.responses.response import Response


class ResponseEvent(ResponseBase):
    type: Literal[
        'response.created',
        'response.in_progress',
        'response.done',
        'response.completed'
    ]
    response: Response

class ResponseOutputItemEvent(ResponseBase):
    type: Literal[
        'response.output_item.added',
        'response.output_item.done'
    ]
    item: Dict[str, Any]
    status: Optional[str] = None


class ResponseReasoningEvent(ResponseBase):
    type: Literal[
        'response.reasoning_summary_part.added',
        'response.reasoning_summary_part.done',
    ]
    item_id: str

class ResponseReasoningDeltaEvent(ResponseBase):
    type: Literal[
        'response.reasoning_summary_text.created',
        'response.reasoning_summary_text.delta',
        'response.reasoning_summary_text.done']
    delta: Optional[str] = None
    item_id: str


class ResponseToolCallEvent(ResponseBase):
    type: str
    item_id: str
    status: Optional[str] = None

    @model_validator(mode="after")
    def infer_status(self) -> "ResponseToolCallEvent":
        if "searching" in self.type:
            self.status = "searching"
        elif "completed" in self.type:
            self.status = "completed"
        return self

class ResponseContentEvent(ResponseBase):
    type: Literal[
        'response.content_part.added',
        'response.content_part.done']
    item_id: str
    logprobs: Optional[List[Logprob]] = Field(default_factory=list)
    text: Optional[str] = None

class ResponseContentDeltaEvent(ResponseBase):
    type: Literal[
    'response.output_text.created',
    'response.output_text.delta',
    'response.output_text.done']
    delta: Optional[str] = None
    item_id: str
    logprobs: Optional[List[Logprob]] = Field(default_factory=list)


AnyResponseEvent = Union[
        ResponseOutputItemEvent,
        ResponseReasoningEvent,
        ResponseReasoningDeltaEvent,
        ResponseContentEvent,
        ResponseContentDeltaEvent,
        ResponseToolCallEvent,
        ResponseEvent
    ]

class ResponseEventRoot(RootModel):
    root: AnyResponseEvent
