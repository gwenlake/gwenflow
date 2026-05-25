from dataclasses import dataclass
from typing import Any

from gwenflow.logger import logger
from gwenflow.memory.base import BaseChatMemory
from gwenflow.types import Message, TextContent
from gwenflow.utils.tokens import keep_tokens_from_text

DEFAULT_TOKEN_LIMIT = 8192
DEFAULT_TOKEN_LIMIT_RATIO = 0.75
MAX_MESSAGE_CONTENT = 0.5


@dataclass
class ChatMemoryBuffer(BaseChatMemory):
    """Sliding-window memory that keeps the most recent messages within a token budget.

    Two-pass strategy:
      1. Pre-filter: any single message exceeding `MAX_MESSAGE_CONTENT * token_limit`
         has its TEXT parts truncated. Multi-modal parts (image/audio/file) and
         thinking_parts are left intact — clamping them would corrupt the wire
         format. If a single message remains over budget after truncation, we
         accept it and let the prune loop drop older messages.
      2. Prune: drop oldest messages until total tokens fit. Never starts the
         kept window with a `tool` message (orphan response) or with an
         `assistant` message whose tool_calls would lose their responses.
    """

    token_limit: int | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.token_limit:
            self.token_limit = int(DEFAULT_TOKEN_LIMIT * DEFAULT_TOKEN_LIMIT_RATIO)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _clamp_message_text(self, message: Message, max_tokens: int) -> None:
        """Truncate the TEXT portion of a message in place to fit `max_tokens`.

        Multi-modal parts, tool_calls, and thinking_parts are preserved as-is —
        we only shrink free-form text. If content is a list of parts, only
        TextContent items are clamped (each independently to the same budget).
        """
        if isinstance(message.content, str):
            message.content = keep_tokens_from_text(
                message.content,
                token_limit=max_tokens,
                tokenizer_fn=self.tokenizer_fn,
            )
        elif isinstance(message.content, list):
            new_parts: list[Any] = []
            for p in message.content:
                if isinstance(p, TextContent):
                    clamped = keep_tokens_from_text(
                        p.content,
                        token_limit=max_tokens,
                        tokenizer_fn=self.tokenizer_fn,
                    )
                    new_parts.append(TextContent(content=clamped))
                else:
                    new_parts.append(p)
            message.content = new_parts
        # If content is None, nothing to clamp (the message might be an
        # assistant turn carrying only tool_calls — leave it alone).

    def _initial_token_count(self) -> int:
        if not self.system_prompt:
            return 0
        return self._token_count_for_messages([Message(role="system", content=self.system_prompt)])

    def _prepend_system(self, messages: list[Message]) -> list[Message]:
        if not self.system_prompt:
            return messages
        return [Message(role="system", content=self.system_prompt)] + messages

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def get(self) -> list[Message]:
        initial_token_count = self._initial_token_count()
        if initial_token_count > self.token_limit:
            raise ValueError("Initial token count exceeds token limit")

        chat_history = self.messages
        if not chat_history:
            return self._prepend_system([])

        # Pass 1: clamp very large individual messages (text-only effect)
        per_message_limit = int(MAX_MESSAGE_CONTENT * self.token_limit)
        for index in range(len(chat_history)):
            if self._token_count_for_message(chat_history[index]) > per_message_limit:
                self._clamp_message_text(chat_history[index], max_tokens=per_message_limit)

        # Pass 2: drop oldest until under budget, keeping tool/assistant chains intact
        keep = len(chat_history)
        token_count = self._token_count_for_messages(chat_history) + initial_token_count

        while token_count > self.token_limit and keep > 0:
            keep -= 1
            # Don't start the kept window with an orphan tool/assistant
            while keep > 0 and chat_history[-keep].role in ("tool", "assistant"):
                keep -= 1
            if keep == 0:
                break
            token_count = (
                self._token_count_for_messages(chat_history[-keep:]) + initial_token_count
            )

        if token_count > self.token_limit:
            logger.warning("Token limit exceeded.")
            return self._prepend_system([])

        return self._prepend_system(chat_history[-keep:] if keep > 0 else [])
