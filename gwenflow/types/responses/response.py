from typing import Any, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator

from gwenflow.types.responses import ReasoningItem
from gwenflow.types.usage import Usage


class ResponseReasoningItem(BaseModel):
    id: str
    type: Literal['reasoning']
    summary: Optional[str] = None

    @field_validator('summary', mode='before')
    @classmethod
    def extract_summary_text(cls, v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, list) and len(v) > 0:
            item = v[0]
            if isinstance(item, dict):
                return item.get("text")
            return getattr(item, "text", None)

        if isinstance(v, str):
            return v
        return None


class ResponseToolCallItem(BaseModel):
    id: str
    status: Optional[Literal['in_progress', 'searching', 'completed']] = None
    type: Optional[str] = None


class ResponseContentItem(BaseModel):
    id: str
    type: Literal['message']
    content: Optional[str] = None

    @field_validator('content', mode='before')
    @classmethod
    def extract_content_text(cls, v: Any) -> Optional[str]:
        if v is None:
            return None

        if isinstance(v, list) and len(v) > 0:
            first_item = v[0]

            if hasattr(first_item, 'text'):
                return first_item.text

            if isinstance(first_item, dict):
                return first_item.get('text')
        if isinstance(v, str):
            return v

        return None


class Response(BaseModel):
    id: str
    created_at: float
    completed_at: Optional[float] = None
    object: Literal["response"]
    model: str
    status: Literal['in_progress', 'completed', 'incomplete']
    incomplete_details: Optional[str] = None
    usage: Usage
    output: List[Union[ResponseReasoningItem, ResponseContentItem, ResponseToolCallItem]] = Field(default_factory=list)
    reasoning: Optional[ReasoningItem] = None
    text_format: Optional[Any] = None
    tools: Optional[List[Any]] = None
    tool_choice: Optional[Literal['none', 'auto', 'required']] = None
    temperature: Optional[float] = None
    top_p : Optional[float] = None

    def get_text(self) -> str:
        for item in self.output:
            if isinstance(item, ResponseContentItem) and item.content:
                return item.content
        return ""

    def get_reasoning(self) -> str:
        for item in self.output:
            if isinstance(item, ResponseReasoningItem) and item.summary:
                return item.summary
        return ""
