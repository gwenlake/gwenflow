
from pydantic import BaseModel, Field
from typing import Optional, Union, List
from typing_extensions import Literal
from uuid import uuid4
from datetime import datetime


class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = None

class Choice(BaseModel):
    index: Optional[int] = 0 
    message: ChatMessage
    finish_reason: Optional[str] = "stop"

class ChoiceDelta(BaseModel):
    index: Optional[int] = 0 
    delta: ChatMessage
    finish_reason: Optional[str] = None

class Usage(BaseModel):
    prompt_tokens: Optional[int] = 0
    completion_tokens: Optional[int] = 0
    total_tokens: Optional[int] = 0

class ChatCompletion(BaseModel):
    id: str = Field(default_factory=lambda: "chatcmpl-" + str(uuid4()))
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(round(datetime.now().timestamp())))
    model: Optional[str] = None
    system_fingerprint: Optional[str] = None
    choices: list[Choice]
    usage: Optional[Usage] = None

class ChatCompletionChunk(BaseModel):
    id: str = Field(default_factory=lambda: "chatcmpl-" + str(uuid4()))
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(round(datetime.now().timestamp())))
    model: Optional[str] = None
    system_fingerprint: Optional[str] = None
    choices: list[ChoiceDelta]

class Embedding(BaseModel):
    embedding: List[float]
    index: int
    object: Literal["embedding"]

class EmbeddingResponse(BaseModel):
    data: List[Embedding]
    model: str
    object: Literal["list"]
    usage: Usage
