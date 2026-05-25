import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Annotated, Any

from pydantic import Discriminator

from gwenflow.types.message import (
    AudioContent,
    ImageContent,
    Message,
    ResponsePart,
    TextContent,
    ThinkingContent,
    ToolCall,
)
from gwenflow.types.usage import AgentUsage, RequestUsage
from gwenflow.utils.utils import now_utc


@dataclass
class ToolResponse:
    """The result of executing one tool call from the agent loop.

    `tool_call_id` matches the originating `ToolCall.id` so the LLM can pair
    requests with responses. `content` is the stringified return value sent
    back to the model; `to_message()` wraps it as a `tool`-role `Message`
    ready to append to the conversation history.
    """

    tool_name: str
    tool_call_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tool_args: dict[str, Any] = field(default_factory=dict[str, Any])
    tool_call_error: bool | None = None
    content: str | None = None
    created_at: datetime = field(default_factory=now_utc)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_message(self) -> Message:
        return Message(
            role="tool",
            tool_call_id=self.tool_call_id,
            content=self.content,
        )


@dataclass
class ModelResponse:
    """The decomposed result of one LLM API call.

    `parts` holds the raw stream of typed pieces (text, thinking, tool calls)
    in the order the model emitted them — preserving provider semantics like
    thinking-before-text. The `.text`, `.thinking`, and `.tool_calls` properties
    are convenience views that re-aggregate parts by kind. `parsed` is set
    when a `response_format` Pydantic model was requested.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parts: Sequence[ResponsePart] = field(default_factory=list)
    parsed: Any | None = None
    finish_reason: str | None = None
    usage: RequestUsage = field(default_factory=RequestUsage)
    created_at: datetime = field(default_factory=now_utc)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def text(self) -> str | None:
        """Get the text in the response."""
        texts: list[str] = []
        last_part: ResponsePart | None = None
        for part in self.parts:
            if isinstance(part, TextContent):
                if isinstance(last_part, TextContent):
                    texts[-1] += part.content
                else:
                    texts.append(part.content)
            last_part = part
        if not texts:
            return None

        return "\n\n".join(texts)

    @property
    def content(self) -> str | None:
        return self.text

    @property
    def thinking(self) -> str | None:
        """Get the thinking in the response."""
        thinking_parts = [part.content for part in self.parts if isinstance(part, ThinkingContent)]
        if not thinking_parts:
            return None
        return "\n\n".join(thinking_parts)

    @property
    def tool_calls(self) -> list[ToolCall]:
        return [part for part in self.parts if isinstance(part, ToolCall)]

    @property
    def audio(self) -> AudioContent | None:
        """Aggregate the streamed audio chunks into a single AudioContent, if any.

        Concatenates the base64 data fragments and joins transcript pieces; keeps
        the metadata (`id`, `expires_at`) from the last chunk that carried them.
        """
        audio_parts = [p for p in self.parts if isinstance(p, AudioContent)]
        if not audio_parts:
            return None
        data = "".join(p.data for p in audio_parts if p.data)
        transcripts = [p.transcript for p in audio_parts if p.transcript]
        extra: dict[str, Any] = {}
        for p in audio_parts:
            if p.extra:
                extra.update(p.extra)
        return AudioContent(
            data=data,
            format=audio_parts[-1].format,
            transcript="".join(transcripts) if transcripts else None,
            extra=extra,
        )

    @property
    def images(self) -> list[ImageContent]:
        """Images the model produced (e.g. via image-generation models)."""
        return [p for p in self.parts if isinstance(p, ImageContent)]


class AgentEventType(str, Enum):
    """Discriminator values for the event stream yielded during an agent run.

    Split into lifecycle (started/completed/error/cancelled) and incremental
    (content/thinking/tool_started/tool_completed). Stream consumers use this
    to route deltas — e.g. render `content` as it arrives, log `thinking`
    separately, surface tool calls in the UI.
    """

    STARTED = "started"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"

    CONTENT = "content"
    THINKING = "thinking"
    TOOL_STARTED = "tool_started"
    TOOL_COMPLETED = "tool_completed"


@dataclass(kw_only=True)
class BaseAgentEvent:
    """Common envelope every event in an agent stream carries.

    `agent_id` + `run_id` let consumers correlate events when multiple agents
    run in parallel (e.g. handoffs in a team). Subclasses override
    `event_type` with the discriminator value from `AgentEventType` and may
    add type-specific fields.
    """

    agent_id: str
    run_id: str
    event_type: str
    content: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=now_utc)


@dataclass(kw_only=True)
class AgentEventStarted(BaseAgentEvent):
    event_type: str = AgentEventType.STARTED.value


@dataclass(kw_only=True)
class AgentEventCompleted(BaseAgentEvent):
    event_type: str = AgentEventType.COMPLETED.value


@dataclass(kw_only=True)
class AgentEventError(BaseAgentEvent):
    event_type: str = AgentEventType.ERROR.value


@dataclass(kw_only=True)
class AgentEventCancelled(BaseAgentEvent):
    event_type: str = AgentEventType.CANCELLED.value


@dataclass(kw_only=True)
class AgentEventContent(BaseAgentEvent):
    event_type: str = AgentEventType.CONTENT.value


@dataclass(kw_only=True)
class AgentEventThinking(BaseAgentEvent):
    event_type: str = AgentEventType.THINKING.value


@dataclass(kw_only=True)
class AgentEventToolStarted(BaseAgentEvent):
    event_type: str = AgentEventType.TOOL_STARTED.value
    tool_call_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tool_name: str | None = None
    tool_args: dict[str, Any] = field(default_factory=dict[str, Any])


@dataclass(kw_only=True)
class AgentEventToolCompleted(BaseAgentEvent):
    event_type: str = AgentEventType.TOOL_COMPLETED.value
    tool_call_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tool_name: str | None = None
    tool_args: dict[str, Any] = field(default_factory=dict[str, Any])


AgentResponseEvent = Annotated[
    AgentEventStarted
    | AgentEventCompleted
    | AgentEventError
    | AgentEventCancelled
    | AgentEventContent
    | AgentEventThinking
    | AgentEventToolStarted
    | AgentEventToolCompleted,
    Discriminator("event_type"),
]


@dataclass
class AgentResponse:
    """The final, aggregated result of one `Agent.run()` call.

    `content` is the assistant's last text output; `parsed` is the validated
    Pydantic instance when `response_model` was set. `reasoning_content`
    accumulates thinking text across every LLM turn in the run.
    `messages` contains the assistant + tool messages produced during the
    loop, and `events` is the full event log (also yielded live by
    `run_stream`). `usage` aggregates tokens, request counts, and tool calls
    over the whole run.
    """

    agent_id: str
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str | None = None
    parsed: Any | None = None
    reasoning_content: str | None = None
    messages: list[Message] = field(default_factory=list)
    events: list[AgentResponseEvent] = field(default_factory=list)
    finish_reason: str | None = None
    usage: AgentUsage = field(default_factory=AgentUsage)
    created_at: datetime = field(default_factory=now_utc)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
