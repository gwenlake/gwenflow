from typing import List, Literal, Optional, Union

from pydantic import Field, RootModel, model_validator

from gwenflow.types.responses.base import Logprob, ResponseBase


class ResponseOutputItemEvent(ResponseBase):
    type: Literal[
        'response.output_item.added',
        'response.output_item.done'
    ]
    item: str
    status: Literal['created', 'done']


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
        ResponseReasoningEvent,
        ResponseReasoningDeltaEvent,
        ResponseContentEvent,
        ResponseContentDeltaEvent,
        ResponseToolCallEvent,
    ]

class ResponseEvent(RootModel):
    root: AnyResponseEvent
