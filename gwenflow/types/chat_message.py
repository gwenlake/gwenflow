
from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional, Union, List, Any, Dict, Generator
from typing_extensions import Literal
from uuid import uuid4
from datetime import datetime


class CompletionUsage(BaseModel):
    prompt_tokens: int
    """The number of tokens used by the prompt."""

    completion_tokens: int
    """Number of tokens in the generated completion."""

    total_tokens: int
    """The total number of tokens used by the request."""


class MessageRole(str, Enum):
    """Message role."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"
    CHATBOT = "chatbot"
    MODEL = "model"


# class ChatMessage(BaseModel):
#     role: str
#     content: Optional[str] = None

class ChatMessage(BaseModel):
    """Chat message."""

    role: MessageRole = MessageRole.USER
    content: Optional[Any] = ""
    additional_kwargs: dict = Field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.role.value}: {self.content}"

    @classmethod
    def from_str(
        cls,
        content: str,
        role: Union[MessageRole, str] = MessageRole.USER,
        **kwargs: Any,
    ) -> "ChatMessage":
        if isinstance(role, str):
            role = MessageRole(role)
        return cls(role=role, content=content, **kwargs)

    def dict(self, **kwargs: Any) -> Dict[str, Any]:
        return self.model_dump(**kwargs)

# class LogProb(BaseModel):
#     """LogProb of a token."""

#     token: str = Field(default_factory=str)
#     logprob: float = Field(default_factory=float)
#     bytes: List[int] = Field(default_factory=list)


# class ChatResponse(BaseModel):
#     """Chat response."""

#     message: ChatMessage
#     raw: Optional[Any] = None
#     delta: Optional[str] = None
#     logprobs: Optional[List[List[LogProb]]] = None
#     additional_kwargs: dict = Field(default_factory=dict)

#     def __str__(self) -> str:
#         return str(self.message)


# ChatResponseGen = Generator[ChatResponse, None, None]


class Choice(BaseModel):
    index: Optional[int] = 0 
    message: ChatMessage
    finish_reason: Optional[str] = "stop"

class ChoiceDelta(BaseModel):
    index: Optional[int] = 0 
    delta: ChatMessage
    finish_reason: Optional[str] = None


class ChatCompletion(BaseModel):
    id: str = Field(default_factory=lambda: "chatcmpl-" + str(uuid4()))
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(round(datetime.now().timestamp())))
    model: Optional[str] = None
    system_fingerprint: Optional[str] = None
    choices: list[Choice]
    usage: Optional[CompletionUsage] = None

class ChatCompletionChunk(BaseModel):
    id: str = Field(default_factory=lambda: "chatcmpl-" + str(uuid4()))
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(round(datetime.now().timestamp())))
    model: Optional[str] = None
    system_fingerprint: Optional[str] = None
    choices: list[ChoiceDelta]
