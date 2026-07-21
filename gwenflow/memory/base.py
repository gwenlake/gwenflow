import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from gwenflow.types import AudioContent, FileContent, ImageContent, Message, TextContent, ThinkingContent
from gwenflow.utils.tokens import num_tokens_from_string

# Rough per-part token cost for non-text inputs. These are deliberately
# conservative (slightly over) so the buffer stays under the budget rather
# than overflowing. Real cost depends on the provider/model.
IMAGE_TOKENS_LOW = 85  # OpenAI low-detail baseline
IMAGE_TOKENS_HIGH = 1100  # safe upper bound for a 2048x2048 high-detail image
AUDIO_TOKENS_PER_SECOND = 50  # OpenAI Realtime/audio rough cost
FILE_TOKENS_DEFAULT = 500  # baseline for a small PDF; refined by data size

# How many base64 chars per audio-second / per kB. Used only when we don't
# have a transcript or duration — better an over-estimate than an under-estimate.
AUDIO_B64_CHARS_PER_SEC = 32000  # ~16kHz mono PCM at base64 expansion ratio


@dataclass
class BaseChatMemory:
    id: str | None = None
    system_prompt: str | None = None
    messages: list[Message] = field(default_factory=list)
    tokenizer_fn: Callable | None = None

    def __post_init__(self) -> None:
        if self.id is None:
            self.id = str(uuid.uuid4())
        if self.tokenizer_fn is None:
            self.tokenizer_fn = num_tokens_from_string

    def reset(self):
        self.messages = []

    def get_all(self):
        messages = []
        if self.system_prompt:
            messages.append(Message(role="system", content=self.system_prompt))
        messages.extend(self.messages)
        return messages

    def add_message(self, message):
        if isinstance(message, Message):
            self.messages.append(message)
        elif isinstance(message, dict):
            self.messages.append(Message(**message))
        else:
            self.messages.append(Message(**message.__dict__))

    def add_messages(self, messages: list[Message]):
        for message in messages:
            self.add_message(message)

    def _token_count_for_part(self, part: Any) -> int:
        """Estimate tokens for one MessageContent part."""
        if isinstance(part, str):
            return self.tokenizer_fn(part)
        if isinstance(part, TextContent):
            return self.tokenizer_fn(part.content)
        if isinstance(part, ThinkingContent):
            return self._token_count_for_thinking(part)
        if isinstance(part, ImageContent):
            return IMAGE_TOKENS_HIGH if part.detail == "high" else IMAGE_TOKENS_LOW
        if isinstance(part, AudioContent):
            if part.transcript:
                return self.tokenizer_fn(part.transcript)
            if part.data:
                seconds = max(1, len(part.data) / AUDIO_B64_CHARS_PER_SEC)
                return int(seconds * AUDIO_TOKENS_PER_SECOND)
            return AUDIO_TOKENS_PER_SECOND
        if isinstance(part, FileContent):
            if part.data:
                # ~1 token per 4 base64 chars (the binary that base64 encodes)
                return max(FILE_TOKENS_DEFAULT, len(part.data) // 4)
            return FILE_TOKENS_DEFAULT
        if isinstance(part, dict):
            return self.tokenizer_fn(json.dumps(part))
        return self.tokenizer_fn(str(part))

    def _token_count_for_thinking(self, part: ThinkingContent) -> int:
        """Estimate tokens for one thinking part, `extra` included.

        Anthropic's block signature lives in `extra` and is echoed back verbatim
        on tool-use turns — it is a large base64 blob, not free.
        """
        total = self.tokenizer_fn(part.content)
        if part.extra:
            total += self.tokenizer_fn(json.dumps(part.extra, default=str))
        return total

    def _token_count_for_message(self, m: Message) -> int:
        """Estimate tokens for an entire Message.

        Includes every field the provider will end up sending: content (string
        or parts), tool_calls, tool_call_id, thinking_parts, and the small
        per-message envelope (role/name).
        """
        total = 4  # role + envelope overhead
        if m.name:
            total += self.tokenizer_fn(m.name)
        if m.tool_call_id:
            total += self.tokenizer_fn(m.tool_call_id)
        if isinstance(m.content, list):
            for part in m.content:
                total += self._token_count_for_part(part)
        elif m.content:
            total += self.tokenizer_fn(m.content)
        if m.tool_calls:
            for tc in m.tool_calls:
                total += self.tokenizer_fn(json.dumps(tc, default=str))
        if m.thinking_parts:
            for tp in m.thinking_parts:
                total += self._token_count_for_thinking(tp)
        return total

    def _token_count_for_messages(self, messages: list[Message]) -> int:
        if not messages:
            return 0
        return sum(self._token_count_for_message(m) for m in messages)
