
from pydantic import BaseModel, Field
from typing import Optional, Union, List, Any, Dict, Generator
from typing_extensions import Literal



class CompletionUsage(BaseModel):
    prompt_tokens: int
    """The number of tokens used by the prompt."""

    completion_tokens: int
    """Number of tokens in the generated completion."""

    total_tokens: int
    """The total number of tokens used by the request."""


class Function(BaseModel):
    arguments: str
    name: str
    """The name of the function to call."""


class ChatCompletionMessageToolCall(BaseModel):
    id: str
    """The ID of the tool call."""

    function: Function
    """The function that the model called."""

    type: Literal["function"]
    """The type of the tool. Currently, only `function` is supported."""


class ChatCompletionAudio(BaseModel):
    id: str
    """Unique identifier for this audio response."""

    data: str
    """
    Base64 encoded audio bytes generated by the model, in the format specified in
    the request.
    """

    expires_at: int
    """
    The Unix timestamp (in seconds) for when this audio response will no longer be
    accessible on the server for use in multi-turn conversations.
    """

    transcript: str
    """Transcript of the audio generated by the model."""


class ChatCompletionMessage(BaseModel):
    content: Optional[str] = None
    """The contents of the message."""

    refusal: Optional[str] = None
    """The refusal message generated by the model."""

    role: Literal["assistant"]
    """The role of the author of this message."""

    audio: Optional[ChatCompletionAudio] = None
    """
    If the audio output modality is requested, this object contains data about the
    audio response from the model.
    """

    tool_calls: Optional[List[ChatCompletionMessageToolCall]] = None
    """The tool calls generated by the model, such as function calls."""



# class ChoiceLogprobs(BaseModel):
#     content: Optional[List[ChatCompletionTokenLogprob]] = None
#     """A list of message content tokens with log probability information."""

#     refusal: Optional[List[ChatCompletionTokenLogprob]] = None
#     """A list of message refusal tokens with log probability information."""


class Choice(BaseModel):
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter", "function_call"]
    """The reason the model stopped generating tokens.

    This will be `stop` if the model hit a natural stop point or a provided stop
    sequence, `length` if the maximum number of tokens specified in the request was
    reached, `content_filter` if content was omitted due to a flag from our content
    filters, `tool_calls` if the model called a tool, or `function_call`
    (deprecated) if the model called a function.
    """

    index: int
    """The index of the choice in the list of choices."""

    # logprobs: Optional[ChoiceLogprobs] = None
    # """Log probability information for the choice."""

    message: ChatCompletionMessage
    """A chat completion message generated by the model."""


class ChatCompletion(BaseModel):
    id: str
    """A unique identifier for the chat completion."""

    choices: List[Choice]
    """A list of chat completion choices.

    Can be more than one if `n` is greater than 1.
    """

    created: int
    """The Unix timestamp (in seconds) of when the chat completion was created."""

    model: str
    """The model used for the chat completion."""

    object: Literal["chat.completion"]
    """The object type, which is always `chat.completion`."""

    service_tier: Optional[Literal["scale", "default"]] = None
    """The service tier used for processing the request.

    This field is only included if the `service_tier` parameter is specified in the
    request.
    """

    system_fingerprint: Optional[str] = None
    """This fingerprint represents the backend configuration that the model runs with.

    Can be used in conjunction with the `seed` request parameter to understand when
    backend changes have been made that might impact determinism.
    """

    usage: Optional[CompletionUsage] = None
    """Usage statistics for the completion request."""
