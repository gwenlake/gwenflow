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

    Three-pass strategy:
      1. Pre-filter: any single message exceeding `MAX_MESSAGE_CONTENT * token_limit`
         has its TEXT parts truncated. Multi-modal parts (image/audio/file) and
         thinking_parts are left intact — clamping them would corrupt the wire
         format. If a single message remains over budget after truncation, we
         accept it and let the prune loop drop older messages.
      2. Prune: keep the longest suffix of the history that fits the budget.
      3. Anchor: the live turn — the last `user` message — is always kept, even
         when the suffix that fits no longer reaches it (a long tool loop can
         easily blow the budget on its own). Without this the model would be
         called with a system prompt and no task at all. When the turn overflows,
         its oldest messages are dropped instead of the user question.

    The window never starts on a `tool` message: its `tool_calls` parent would
    have been pruned, leaving an orphan response the providers reject.
    """

    token_limit: int | None = None

    reserved_tokens: int = 0
    """Tokens the request spends outside the message list — the tool schemas the
    provider sends alongside every call. Not knowing about them made the buffer
    believe it was under budget while the request overflowed. The agent refreshes
    this at the start of each run."""

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
        we only shrink free-form text. If content is a list of parts, the budget
        is split evenly across the TextContent items so that a message with many
        text parts still lands under `max_tokens`.
        """
        if isinstance(message.content, str):
            message.content = keep_tokens_from_text(
                message.content,
                token_limit=max_tokens,
                tokenizer_fn=self.tokenizer_fn,
            )
        elif isinstance(message.content, list):
            text_parts = sum(1 for p in message.content if isinstance(p, TextContent))
            per_part_limit = max_tokens // text_parts if text_parts else max_tokens
            new_parts: list[Any] = []
            for p in message.content:
                if isinstance(p, TextContent):
                    clamped = keep_tokens_from_text(
                        p.content,
                        token_limit=per_part_limit,
                        tokenizer_fn=self.tokenizer_fn,
                    )
                    new_parts.append(TextContent(content=clamped))
                else:
                    new_parts.append(p)
            message.content = new_parts
        # If content is None, nothing to clamp (the message might be an
        # assistant turn carrying only tool_calls — leave it alone).

    def _last_user_index(self, chat_history: list[Message]) -> int | None:
        """Index of the most recent `user` message — the start of the live turn."""
        for index in range(len(chat_history) - 1, -1, -1):
            if chat_history[index].role == "user":
                return index
        return None

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

        budget = self.token_limit - initial_token_count - self.reserved_tokens
        if budget <= 0:
            logger.warning(
                f"System prompt and tool schemas fill the whole context "
                f"({initial_token_count + self.reserved_tokens} of {self.token_limit} tokens): "
                f"no room left for the conversation."
            )
            budget = 0

        # Pass 1: clamp very large individual messages (text-only effect)
        per_message_limit = int(MAX_MESSAGE_CONTENT * budget)
        for message in chat_history:
            if self._token_count_for_message(message) > per_message_limit:
                self._clamp_message_text(message, max_tokens=per_message_limit)

        counts = [self._token_count_for_message(m) for m in chat_history]

        # Pass 2: keep the longest suffix that fits the budget
        start = len(chat_history)
        token_count = 0
        for index in range(len(chat_history) - 1, -1, -1):
            if token_count + counts[index] > budget:
                break
            token_count += counts[index]
            start = index

        # Pass 3: never drop the live turn. If the suffix that fits stops short of
        # the last user message, pull that message back in and drop the oldest
        # messages of the turn instead — a task-less prompt is worse than a
        # truncated one.
        anchor = self._last_user_index(chat_history)
        if anchor is not None and start > anchor:
            token_count += counts[anchor]
            while token_count > budget and start < len(chat_history) - 1:
                token_count -= counts[start]
                start += 1

        # An orphan `tool` message (its `tool_calls` parent was pruned) is
        # rejected by the providers: skip forward past any leading one.
        while start < len(chat_history) and chat_history[start].role == "tool":
            token_count -= counts[start]
            start += 1

        kept = chat_history[start:]
        if anchor is not None and anchor < start:
            kept = [chat_history[anchor]] + kept

        if not kept:
            logger.warning("Token limit exceeded: no message left to send.")
        elif token_count > budget:
            logger.warning(f"Token limit exceeded: sending the current turn anyway ({token_count} > {budget} tokens).")

        return self._prepend_system(kept)
