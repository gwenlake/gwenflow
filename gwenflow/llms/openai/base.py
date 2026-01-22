from typing import Annotated, Any, List, Literal, Optional, Union
from gwenflow.types import Usage
from gwenflow.tools import BaseTool
from pydantic import BaseModel, Field, RootModel


class ResponseContent(BaseModel):
    type: Literal['output_text', 'summary_text']
    text: Any
    annotations: Optional[List] = Field(default_factory=list)


class ResponseOutputItem(BaseModel):
    id: str
    type: Literal["reasoning", "message"]
    status: Optional[str] = None
    summary: Optional[List[ResponseContent]] = None
    content: Optional[List[ResponseContent]] = None
    role: Optional[str] = None


class ReasoningItem(BaseModel):
    effort: Literal['low', 'medium', 'high']
    summary: Literal['auto', 'concise', 'detailed']


class Response(BaseModel):
    model_config = {"extra": "ignore"}
    id: str
    object: Literal["response"]
    model: str
    status: str
    created_at: float
    completed_at: Optional[float] = None
    output: List[ResponseOutputItem]
    usage: Usage
    reasoning: Optional[ReasoningItem] = None
    text_format: Optional[Any] = None
    tools: Optional[List[BaseTool]] = None
    tool_choice: Optional[Literal['none', 'auto', 'required']] = None
    temperature: Optional[float] = None
    top_p : Optional[float] = None

    def get_text(self) -> str:
        for item in self.output:
            if item.type == "message" and item.content:
                return item.content[0].text
        return ""

    def get_reasoning(self) -> str:
        for item in self.output:
            if item.type == "reasoning" and item.summary:
                return item.summary[0].text
        return ""


class Logprob(BaseModel):
    token: str
    logprob: float
    bytes: Optional[List[int]] = None

class ResponseDeltaEventBase(BaseModel):
    item_id: str
    sequence_number: int


class ResponseReasoningDeltaEvent(ResponseDeltaEventBase):
    type: Literal["response.reasoning_summary_text.delta"]
    delta: str

class ResponseTextDeltaEvent(ResponseDeltaEventBase):
    type: Literal["response.output_text.delta"]
    delta: str
    logprobs: Optional[List[Logprob]] = Field(default_factory=list)


class ResponseDoneEvent(BaseModel):
    type: Literal["response.completed"]
    response: Response

AnyResponseEvent = Annotated[
    Union[
        ResponseReasoningDeltaEvent,
        ResponseTextDeltaEvent,
        ResponseDoneEvent
    ],
    Field(discriminator="type")
]

class ResponseEvent(RootModel):
    root: AnyResponseEvent
