import copy
import json
from dataclasses import dataclass
from typing import Any

from gwenflow.logger import logger
from gwenflow.memory.base import BaseChatMemory
from gwenflow.types import Message, TextContent
from gwenflow.utils.tokens import keep_tokens_from_text

DEFAULT_TOKEN_LIMIT = 8192
DEFAULT_TOKEN_LIMIT_RATIO = 0.75
MAX_MESSAGE_CONTENT = 0.5

# Calibration of the local token estimate against the provider's real count.
MIN_TOKEN_SCALE = 0.7
MAX_TOKEN_SCALE = 4.0
MIN_CALIBRATION_TOKENS = 256
TOKEN_SCALE_DECAY = 0.8

CLEARED_TOOL_RESULT = (
    "[Result of {call} removed to free context ({tokens} tokens). Call the tool again if you still need it.]"
)
DUPLICATE_TOOL_RESULT = "[Same call as a later {call}: result removed, see the more recent one.]"


@dataclass
class ChatMemoryBuffer(BaseChatMemory):
    """Sliding-window memory that keeps the most recent messages within a token budget.

    Four-pass strategy (each pass only runs if the previous ones did not fit):
      0. Lighten tool results — cheap and model-free. Results of a call that is
         repeated later (same tool, same arguments) are replaced by a stub; then
         the oldest results outside the `keep_recent_tool_results` most recent
         are replaced by a one-line stub, oldest first, until the window fits.
         The call itself stays visible, so the model knows it can call again.
      1. Pre-filter: any single message exceeding `MAX_MESSAGE_CONTENT * token_limit`
         has its TEXT parts truncated — on a copy, never on the stored history.
         Multi-modal parts (image/audio/file) and thinking_parts are left intact —
         clamping them would corrupt the wire format. If a single message remains
         over budget after truncation, we accept it and let the prune loop drop
         older messages.
      2. Prune: keep the longest suffix of the history that fits the budget.
      3. Anchor: the live turn — the last `user` message — is always kept, even
         when the suffix that fits no longer reaches it (a long tool loop can
         easily blow the budget on its own). Without this the model would be
         called with a system prompt and no task at all. When the turn overflows,
         its oldest messages are dropped instead of the user question.

    The window never starts on a `tool` message: its `tool_calls` parent would
    have been pruned, leaving an orphan response the providers reject.

    `get()` is a pure read: it never mutates the stored history and never calls
    an LLM. Compaction of evicted messages is a separate command — `compact()` /
    `acompact()` — that is a no-op here and is implemented by
    `ChatSummaryMemoryBuffer`. The agent calls it before each `get()`.
    """

    token_limit: int | None = None

    reserved_tokens: int = 0
    """Tokens the request spends outside the message list — the tool schemas the
    provider sends alongside every call. Not knowing about them made the buffer
    believe it was under budget while the request overflowed. The agent refreshes
    this at the start of each run."""

    token_scale: float = 1.0
    """Ratio between the provider's real prompt token count and the local
    estimate (tiktoken's cl100k_base, which under-counts for the Mistral, Qwen
    or Gemma tokenizers, especially on French, JSON or code). The agent feeds it
    through `observe_prompt_tokens()` after each call; the budget is divided by it."""

    keep_recent_tool_results: int = 3
    """Tool results that pass 0 never clears (the most recent ones)."""

    min_cleared_tool_result_tokens: int = 200
    """Tool results smaller than this are not worth clearing."""

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
    # Calibration against the provider's token count
    # ------------------------------------------------------------------

    def _effective_token_limit(self) -> int:
        """`token_limit` expressed in local-estimate tokens."""
        return int(self.token_limit / self.token_scale)

    def observe_prompt_tokens(self, window: list[Message], actual_tokens: int) -> None:
        """Calibrate the local estimate with the prompt tokens the provider billed.

        `window` is the list `get()` returned for that call (system prompt
        included); the tool schemas are added through `reserved_tokens`. An
        under-estimate is corrected at once (the next call must not overflow);
        an over-estimate is relaxed slowly.
        """
        if not actual_tokens or actual_tokens <= 0:
            return
        estimated = self._token_count_for_messages(window) + self.reserved_tokens
        if estimated < MIN_CALIBRATION_TOKENS:
            return  # too small to say anything about the tokenizer
        ratio = min(max(actual_tokens / estimated, MIN_TOKEN_SCALE), MAX_TOKEN_SCALE)
        if ratio > self.token_scale:
            self.token_scale = ratio
        else:
            self.token_scale = TOKEN_SCALE_DECAY * self.token_scale + (1 - TOKEN_SCALE_DECAY) * ratio

    # ------------------------------------------------------------------
    # Pass 0: lighten tool results
    # ------------------------------------------------------------------

    @staticmethod
    def _tool_calls_by_id(chat_history: list[Message]) -> dict[str, tuple[str, str]]:
        """Map each tool_call_id to `(tool name, arguments)` from the assistant turns."""
        calls: dict[str, tuple[str, str]] = {}
        for message in chat_history:
            for tool_call in message.tool_calls or []:
                function = tool_call.get("function") or {}
                name = function.get("name") or tool_call.get("name") or "tool"
                arguments = function.get("arguments", tool_call.get("arguments", ""))
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments, sort_keys=True, default=str)
                if tool_call.get("id"):
                    calls[tool_call["id"]] = (name, arguments)
        return calls

    @staticmethod
    def _describe_call(name: str, arguments: str, max_chars: int = 120) -> str:
        if len(arguments) > max_chars:
            arguments = arguments[:max_chars] + "..."
        return f"{name}({arguments})"

    def _stub(self, message: Message, text: str) -> Message:
        stub = copy.copy(message)
        stub.content = text
        return stub

    def _lighten_tool_results(self, chat_history: list[Message], budget: int) -> list[Message]:
        """Pass 0: replace dispensable tool results by stubs, on copies.

        Runs only when the history does not fit. First the results of calls
        repeated later with the same arguments (a file read twice...), then the
        oldest results beyond the `keep_recent_tool_results` most recent, oldest
        first, until the history fits. Never touches the stored history.
        """
        counts = [self._token_count_for_message(m) for m in chat_history]
        total = sum(counts)
        if total <= budget:
            return chat_history

        calls = self._tool_calls_by_id(chat_history)
        tool_indexes = [i for i, m in enumerate(chat_history) if m.role == "tool" and m.tool_call_id in calls]

        def replace(index: int, text: str) -> None:
            nonlocal total
            stub = self._stub(chat_history[index], text)
            stub_count = self._token_count_for_message(stub)
            total -= counts[index] - stub_count
            counts[index] = stub_count
            chat_history[index] = stub

        # a) duplicates: keep only the most recent result of an identical call
        seen: set[tuple[str, str]] = set()
        for index in reversed(tool_indexes):
            key = calls[chat_history[index].tool_call_id]
            if key in seen and counts[index] >= self.min_cleared_tool_result_tokens:
                replace(index, DUPLICATE_TOOL_RESULT.format(call=self._describe_call(*key)))
            seen.add(key)

        # b) old results, oldest first, until the history fits
        keep = max(self.keep_recent_tool_results, 0)
        candidates = tool_indexes[:-keep] if keep else tool_indexes
        for index in candidates:
            if total <= budget:
                break
            if counts[index] < self.min_cleared_tool_result_tokens:
                continue
            name, arguments = calls[chat_history[index].tool_call_id]
            replace(
                index,
                CLEARED_TOOL_RESULT.format(call=self._describe_call(name, arguments), tokens=counts[index]),
            )
        return chat_history

    # ------------------------------------------------------------------
    # Window computation (shared by get() and subclass compaction)
    # ------------------------------------------------------------------

    def _budget(self) -> int:
        """Token budget left for the conversation window.

        `token_limit` minus the system prompt, the reserved (tool schema)
        tokens, and the summary message if a subclass maintains one.
        """
        token_limit = self._effective_token_limit()
        initial_token_count = self._initial_token_count()
        if initial_token_count > token_limit:
            raise ValueError("Initial token count exceeds token limit")

        budget = token_limit - initial_token_count - self.reserved_tokens
        summary_message = self._summary_message()
        if summary_message is not None:
            budget -= self._token_count_for_message(summary_message)
        if budget <= 0:
            logger.warning(
                f"System prompt, tool schemas and summary fill the whole context "
                f"({token_limit - budget} of {token_limit} tokens): "
                f"no room left for the conversation."
            )
            budget = 0
        return budget

    def _clamp_history(self, chat_history: list[Message], budget: int) -> list[Message]:
        """Passes 0 and 1: lighten tool results, then clamp large messages.

        Clamping only shortens text. Works on copies: the stored history is never mutated, and a
        non-positive limit (budget exhausted) never wipes every message.
        """
        chat_history = self._lighten_tool_results(chat_history, budget)
        per_message_limit = int(MAX_MESSAGE_CONTENT * budget)
        if per_message_limit <= 0:
            return chat_history
        for index, message in enumerate(chat_history):
            if self._token_count_for_message(message) > per_message_limit:
                clamped = copy.deepcopy(message)
                self._clamp_message_text(clamped, max_tokens=per_message_limit)
                chat_history[index] = clamped
        return chat_history

    def _prune(self, chat_history: list[Message], budget: int) -> tuple[int, int | None, int]:
        """Passes 2 and 3: locate the window.

        Returns `(start, anchor, token_count)` where `start` is the first index
        of the window and `anchor` the live-turn index (may be < start, in
        which case the anchor message is pulled back in by the caller).
        """
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

        return start, anchor, token_count

    # ------------------------------------------------------------------
    # Hooks and commands for subclasses (e.g. ChatSummaryMemoryBuffer)
    # ------------------------------------------------------------------

    def _summary_message(self) -> Message | None:
        """Message summarizing evicted history, sent first in the window.

        The base buffer keeps no summary.
        """
        return None

    def compact(self) -> None:
        """Fold about-to-be-evicted messages into a summary.

        No-op here: the base buffer simply drops evicted messages. The agent
        calls this before each `get()`.
        """

    async def acompact(self) -> None:
        """Async variant of `compact()`. No-op here."""

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def get(self) -> list[Message]:
        """Compute the message window to send to the model.

        A pure read: never mutates the stored history, never calls an LLM.
        """
        budget = self._budget()
        summary_message = self._summary_message()

        chat_history = list(self.messages)
        if not chat_history:
            return self._prepend_system([summary_message] if summary_message else [])

        chat_history = self._clamp_history(chat_history, budget)
        start, anchor, token_count = self._prune(chat_history, budget)

        kept = chat_history[start:]
        if anchor is not None and anchor < start:
            kept = [chat_history[anchor]] + kept
        if summary_message is not None:
            kept = [summary_message] + kept

        if not kept:
            logger.warning("Token limit exceeded: no message left to send.")
        elif token_count > budget:
            logger.warning(f"Token limit exceeded: sending the current turn anyway ({token_count} > {budget} tokens).")

        return self._prepend_system(kept)
