import copy
import json
import math
from dataclasses import dataclass, field
from typing import Any

from gwenflow.logger import logger
from gwenflow.memory.base import BaseChatMemory
from gwenflow.memory.compaction import (
    CompactionContext,
    CompactionRecord,
    CompactionStrategy,
    default_strategies,
    protected_indexes,
)
from gwenflow.types import AgentUsage, ContextUsage, Message, TextContent
from gwenflow.utils.tokens import keep_tokens_from_text

DEFAULT_TOKEN_LIMIT = 8192
DEFAULT_TOKEN_LIMIT_RATIO = 0.75
DEFAULT_CONTEXT_WINDOW = 128000
"""Window assumed by a fraction-based buffer until the agent tells it the
model's real one (same value as `gwenflow.llms.base.DEFAULT_CONTEXT_SIZE`)."""
DEFAULT_MAX_FRACTION = 0.85
"""Trigger of the agent's default buffer: compaction starts when the next
request would use this share of the window."""
DEFAULT_TARGET_RATIO = 0.70
"""Default target of `compact()`, as a share of the limit (0.85 * 0.70 = 60 %
of the window for the agent's default buffer). The gap between trigger and
target is what keeps the sent prefix stable, and the prompt cache valid,
between two compaction batches."""
MAX_COMPACTION_RECORDS = 20
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
    """Chat memory that keeps the history within a token budget.

    Two mechanisms, one persistent and one on the fly:

    `compact()` rewrites the stored history in tiers (see
    `gwenflow.memory.compaction`): deduplicate file reads, clamp oversized
    parts, clear old tool results, summarize older messages (with a
    summarizer, see `ChatSummaryMemoryBuffer`), drop the oldest messages.
    It triggers when the next request would exceed `limit()` and stops once
    the history is under `target()`, so it works in batches and the prefix
    sent to the model is identical between two batches. The agent calls it
    before each request. The first user message, the task the agent pinned
    and any message marked with `compaction.pin()` always survive.

    `get()` is a pure read that returns the window to send. When the stored
    history fits (always true right after `compact()`), it is the history as
    is. Otherwise, as a last resort (a buffer used without `compact()`, a
    single message bigger than the budget), the four passes below shrink the
    window on copies, without touching the stored history:
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
    """Absolute budget of the request, in tokens (system prompt, tool schemas
    and messages included). Exclusive with `max_fraction`. Without either,
    `DEFAULT_TOKEN_LIMIT * DEFAULT_TOKEN_LIMIT_RATIO`."""

    max_fraction: float | None = None
    """Budget as a share of the model's context window, resolved on every
    request against `context_window` (the agent refreshes it from the LLM it
    actually calls, so swapping `agent.llm` moves the budget). Exclusive with
    `token_limit`."""

    context_window: int | None = None
    """Context window of the model the next request goes to, in tokens. Set
    by the agent before each request; `DEFAULT_CONTEXT_WINDOW` until then."""

    context_window_resolved: bool = True
    """Whether `context_window` is known (table, server, explicit) or assumed
    (`fallback_context_window`, library default). Reported by `context_usage()`."""

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

    _anchor_tokens: int | None = None
    """Prompt + completion tokens the provider reported for the last request,
    or None when no request has been observed since the history changed shape."""

    _anchor_index: int = 0
    """Number of stored messages the anchor accounts for: those were in the
    last request (as sent) or are its reply. Messages past it are estimated."""

    target_fraction: float | None = None
    """Where `compact()` stops, as a share of the context window. Must be
    below `max_fraction`. None: `DEFAULT_TARGET_RATIO` of the limit."""

    target_tokens: int | None = None
    """Where `compact()` stops, in tokens (absolute). None: `DEFAULT_TARGET_RATIO`
    of the limit. Exclusive with `target_fraction`."""

    strategies: list[CompactionStrategy] | None = None
    """Compaction tiers, cheapest first. None: `default_strategies()`, with
    the summarizer of a `ChatSummaryMemoryBuffer`."""

    compactions: list[CompactionRecord] = field(default_factory=list)
    """What the last compactions did (most recent last, capped)."""

    usage: AgentUsage = field(default_factory=AgentUsage)
    """Cumulative usage of the LLM calls made by compaction (summaries).
    Deliberately NOT cleared by `reset()`: tokens already spent stay
    accounted for (see /cost)."""

    _pending_usage: AgentUsage = field(default_factory=AgentUsage)
    """Compaction usage not yet folded into a run (see `drain_usage`)."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.token_limit is not None and self.max_fraction is not None:
            raise ValueError("token_limit and max_fraction are exclusive: pass one of them.")
        if self.max_fraction is not None and not 0 < self.max_fraction <= 1:
            raise ValueError(f"max_fraction must be in (0, 1], got {self.max_fraction}.")
        if self.target_fraction is not None and self.target_tokens is not None:
            raise ValueError("target_fraction and target_tokens are exclusive: pass one of them.")
        if self.token_limit is None and self.max_fraction is None:
            self.token_limit = int(DEFAULT_TOKEN_LIMIT * DEFAULT_TOKEN_LIMIT_RATIO)
        if self.target() >= self.limit():
            raise ValueError(f"the compaction target ({self.target()}) must be below the limit ({self.limit()}).")

    # ------------------------------------------------------------------
    # Budget
    # ------------------------------------------------------------------

    def limit(self) -> int:
        """The request budget in provider tokens.

        `token_limit` when set, else `max_fraction` of the current `context_window`.
        """
        if self.max_fraction is not None:
            return int((self.context_window or DEFAULT_CONTEXT_WINDOW) * self.max_fraction)
        return int(self.token_limit)

    def target(self) -> int:
        """Where `compact()` stops, in provider tokens (see `target_fraction`, `target_tokens`)."""
        if self.target_tokens is not None:
            return int(self.target_tokens)
        if self.target_fraction is not None:
            return int((self.context_window or DEFAULT_CONTEXT_WINDOW) * self.target_fraction)
        return int(self.limit() * DEFAULT_TARGET_RATIO)

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
        """The budget expressed in local-estimate tokens."""
        return int(self.limit() / self.token_scale)

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

    def observe_usage(self, window: list[Message], input_tokens: int, output_tokens: int = 0) -> None:
        """Anchor the size of the history on what the provider just reported.

        Call it once the reply has been stored: `input_tokens + output_tokens`
        then covers everything in the request (system prompt, tool schemas,
        files, the window) plus the reply, and the messages stored after this
        point are the only ones left to estimate. The same count also
        calibrates `token_scale` for the parts that still need estimating.
        """
        self.observe_prompt_tokens(window, input_tokens)
        if not input_tokens or input_tokens <= 0:
            return
        self._anchor_tokens = int(input_tokens) + int(output_tokens or 0)
        self._anchor_index = len(self.messages)

    def _invalidate_anchor(self) -> None:
        """Forget the anchor: the stored history changed shape (compaction, reset, load)."""
        self._anchor_tokens = None
        self._anchor_index = 0

    def _anchor_valid(self) -> bool:
        return self._anchor_tokens is not None and self._anchor_index <= len(self.messages)

    def _scaled(self, local_tokens: int) -> int:
        return int(local_tokens * self.token_scale)

    def used_tokens(self) -> int:
        """Provider tokens the next request would carry if the history were sent as is.

        Anchored: the last reported usage plus a scaled local estimate of the
        messages added since. Unanchored: a scaled local estimate of the system
        prompt, the tool schemas, the summary and every stored message.
        """
        if self._anchor_valid():
            added = self._token_count_for_messages(self.messages[self._anchor_index :])
            return self._anchor_tokens + self._scaled(added)
        total = self._initial_token_count() + self.reserved_tokens
        summary_message = self._summary_message()
        if summary_message is not None:
            total += self._token_count_for_message(summary_message)
        total += self._token_count_for_messages(self.messages)
        return self._scaled(total)

    def context_usage(self) -> ContextUsage:
        """Where the history stands against the model's window (see `ContextUsage`)."""
        return ContextUsage(
            used_tokens=self.used_tokens(),
            window_tokens=self.context_window or DEFAULT_CONTEXT_WINDOW,
            limit_tokens=self.limit(),
            resolved=self.context_window_resolved,
            anchored=self._anchor_valid(),
        )

    def reset(self):
        super().reset()
        self._invalidate_anchor()

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

    def _token_limit_for_display(self) -> int:
        return self.limit()

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
        # truncated one. Protected messages (first user message, pinned task)
        # are pulled back in the same way by `get()`.
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

    def _summarizer(self) -> Any:
        """LLM used by the summary tier; the base buffer has none."""
        return None

    def _default_strategies(self) -> list[CompactionStrategy]:
        return default_strategies(summarizer=self._summarizer())

    def _compaction_context(self, excess: int) -> CompactionContext:
        return CompactionContext(
            count=self._token_count_for_message,
            tokenizer_fn=self.tokenizer_fn,
            excess=excess,
            budget=self._budget(),
            summary=getattr(self, "summary", "") or "",
        )

    def _store_summary(self, summary: str) -> None:
        """Hook for the summary buffer; the base buffer keeps no summary."""

    def _local_estimate(self, messages: list[Message], summary: str = "") -> int:
        """Local tokens of a request made of `messages` with `summary` in front."""
        total = self._initial_token_count() + self.reserved_tokens + self._token_count_for_messages(messages)
        if summary:
            total += self._token_count_for_message(Message(role="assistant", content=summary)) + 12
        return total

    def compact(self) -> CompactionRecord | None:
        """Rewrite the stored history so that the next request fits.

        Runs the always-on tiers on every call, and the others only when the
        next request would exceed `limit()`, cheapest first, until the history
        is back under `target()`. A tier that raises is skipped (the sliding
        window at the end always succeeds). Returns what was done, or None
        when nothing changed. The agent calls it before each `get()`.
        """
        if not self.messages:
            return None
        tiers = self.strategies if self.strategies is not None else self._default_strategies()
        used = self.used_tokens()
        over = used > self.limit()
        excess = math.ceil((used - self.target()) / self.token_scale) if over else 0
        ctx = self._compaction_context(excess)
        before = list(self.messages)
        messages = before
        applied: list[str] = []
        for tier in tiers:
            if not over and not tier.always:
                continue
            if over and ctx.excess <= 0 and not tier.always:
                break
            try:
                candidate = tier.compact(messages, ctx)
            except Exception as e:  # noqa: BLE001 — a failing tier must not fail the run
                logger.warning(f"Compaction tier '{tier.name}' failed, moving on to the next one: {e}")
                continue
            if candidate is messages:
                continue
            messages = candidate
            applied.append(tier.name)
            # Re-measure (a summary adds its own message to the window).
            ctx.excess = self._local_estimate(messages, ctx.summary) - math.ceil(self.target() / self.token_scale)
        self.usage.add(ctx.usage)
        self._pending_usage.add(ctx.usage)
        summary_changed = ctx.summary != (getattr(self, "summary", "") or "")
        if summary_changed:
            self._store_summary(ctx.summary)
        if not applied and not summary_changed:
            return None
        originals = {id(m) for m in messages}
        record = CompactionRecord(
            strategies=applied,
            messages_before=len(before),
            messages_after=len(messages),
            tokens_before=used,
            tokens_after=0,
            notes=ctx.notes,
            removed=[m for m in before if id(m) not in originals],
        )
        self.messages = messages
        self._invalidate_anchor()
        record.tokens_after = self.used_tokens()
        self.compactions.append(record)
        del self.compactions[:-MAX_COMPACTION_RECORDS]
        return record

    def drain_usage(self) -> AgentUsage:
        """Take the compaction usage accumulated since the last drain.

        The agent folds it into the usage of the run that triggered it, so
        summaries are part of `AgentResponse.usage` and `Agent.total_usage`.
        """
        drained = self._pending_usage
        self._pending_usage = AgentUsage()
        return drained

    async def acompact(self) -> CompactionRecord | None:
        """Async variant of `compact()`: the (sync) LLM calls run in a worker thread."""
        import asyncio

        return await asyncio.to_thread(self.compact)

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
        pulled = protected_indexes(chat_history)
        if anchor is not None:
            pulled.add(anchor)
        kept = [chat_history[index] for index in sorted(pulled) if index < start] + kept
        if summary_message is not None:
            kept = [summary_message] + kept

        if not kept:
            logger.warning("Token limit exceeded: no message left to send.")
        elif token_count > budget:
            logger.warning(f"Token limit exceeded: sending the current turn anyway ({token_count} > {budget} tokens).")

        return self._prepend_system(kept)
