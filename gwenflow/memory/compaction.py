"""Compaction strategies for the chat memory buffers.

Each strategy is a `CompactionStrategy` whose `compact(messages, ctx)` returns
a new list of messages and never mutates the objects it receives: a message
that changes is replaced by a copy. A strategy that has nothing to do returns
the very list it was given, which is how the orchestrator notices.

`ChatMemoryBuffer.compact()` runs the strategies in tiers, cheapest first,
re-measures after each one and stops as soon as the history is back under
its target (see `ChatMemoryBuffer.target()`). The result is persisted in the
stored history, so the prefix sent to the model stays identical between two
compaction batches and the provider's prompt cache keeps working.

Invariants every strategy keeps:

- a tool call and its results form one group that is cleared, clamped,
  summarized or dropped as a whole (see `tool_groups`, `validate_tool_pairs`);
- the first user message of the history, the live task the agent pinned, and
  any message marked with `pin()` are never altered or removed.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from gwenflow.logger import logger
from gwenflow.types import AgentUsage, Message, TextContent

PINNED_KEY = "pinned"
CLAMP_MARKER = "\n[clamped: removed {removed} of {total} characters]\n"
CLEARED_RESULT = (
    "[Result of {call} cleared to free context ({tokens} tokens). Call the tool again if you still need it.]"
)
SUPERSEDED_READ = "[Superseded read of {key}: see the more recent result.]"
DUPLICATE_RESULT = "[Same call as a later {call}: result removed, see the more recent one.]"
CLEARED_ARGUMENTS = "{}"

# ---------------------------------------------------------------------------
# Helpers shared by the strategies
# ---------------------------------------------------------------------------


def pin(message: Message) -> Message:
    """Mark `message` so that no compaction strategy alters or removes it."""
    message.extra[PINNED_KEY] = True
    return message


def unpin(message: Message) -> Message:
    message.extra.pop(PINNED_KEY, None)
    return message


def is_pinned(message: Message) -> bool:
    return bool(message.extra.get(PINNED_KEY))


def tool_groups(messages: list[Message]) -> list[tuple[int, int]]:
    """Atomic ranges `[start, end)` of the history.

    An assistant message carrying tool calls and the `tool` messages that
    follow it form one range; every other message is a range of its own. A
    `tool` message without a preceding call (should not happen) is a range
    of its own too, so the function never fails.
    """
    ranges: list[tuple[int, int]] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        end = index + 1
        if message.role == "assistant" and message.tool_calls:
            while end < len(messages) and messages[end].role == "tool":
                end += 1
        ranges.append((index, end))
        index = end
    return ranges


def validate_tool_pairs(messages: list[Message]) -> None:
    """Raise `ValueError` if a tool call lost its result or a result its call.

    The invariant every provider enforces: an assistant message with tool
    calls is immediately followed by exactly one `tool` message per call id,
    and a `tool` message appears nowhere else.
    """
    index = 0
    while index < len(messages):
        message = messages[index]
        if message.role == "tool":
            raise ValueError(f"orphan tool result at index {index} (tool_call_id={message.tool_call_id!r})")
        if message.role == "assistant" and message.tool_calls:
            expected = [call.get("id") for call in message.tool_calls]
            index += 1
            seen: list[str | None] = []
            while index < len(messages) and messages[index].role == "tool":
                seen.append(messages[index].tool_call_id)
                index += 1
            if sorted(map(str, seen)) != sorted(map(str, expected)):
                raise ValueError(f"tool group mismatch: calls {expected} but results {seen}")
            continue
        index += 1


def first_user_index(messages: list[Message]) -> int | None:
    for index, message in enumerate(messages):
        if message.role == "user":
            return index
    return None


def last_user_index(messages: list[Message]) -> int | None:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].role == "user":
            return index
    return None


def protected_indexes(
    messages: list[Message],
    preserve_first_user_message: bool = True,
    preserve_last_user_message: bool = True,
) -> set[int]:
    """Indexes that must survive compaction.

    Pinned messages, the first user message of the history and the last one
    (the live task): a request without its task is worse than a short one.
    """
    protected = {index for index, message in enumerate(messages) if is_pinned(message)}
    if preserve_first_user_message:
        first = first_user_index(messages)
        if first is not None:
            protected.add(first)
    if preserve_last_user_message:
        last = last_user_index(messages)
        if last is not None:
            protected.add(last)
    return protected


def call_name_and_arguments(call: dict[str, Any]) -> tuple[str, Any]:
    """`(name, arguments)` of a tool call dict, whatever its shape (OpenAI nested or flat)."""
    function = call.get("function") or {}
    name = function.get("name") or call.get("name") or "tool"
    arguments = function.get("arguments", call.get("arguments"))
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (json.JSONDecodeError, TypeError):
            pass
    return name, arguments


def describe_call(name: str, arguments: Any, max_chars: int = 120) -> str:
    rendered = arguments if isinstance(arguments, str) else json.dumps(arguments, sort_keys=True, default=str)
    if rendered is None:
        rendered = ""
    if len(rendered) > max_chars:
        rendered = rendered[:max_chars] + "..."
    return f"{name}({rendered})"


def calls_by_id(messages: list[Message]) -> dict[str, tuple[str, Any]]:
    """Map every tool_call_id of the history to `(name, arguments)`."""
    calls: dict[str, tuple[str, Any]] = {}
    for message in messages:
        for call in message.tool_calls or []:
            if call.get("id"):
                calls[str(call["id"])] = call_name_and_arguments(call)
    return calls


def default_file_key(name: str, arguments: Any) -> str | None:
    """File identity of a read call, for the tools of the coding bundle.

    `ReadFileTool(file_path=...)` (and the usual `read_file` / `path`
    spellings) identify a file; anything else returns None.
    """
    if not isinstance(arguments, dict):
        return None
    if name.lower().replace("_", "") not in {"readfiletool", "readfile", "read", "catfile", "viewfile"}:
        return None
    path = arguments.get("file_path") or arguments.get("path") or arguments.get("filename")
    return f"read:{path}" if path else None


def with_content(message: Message, content: Any) -> Message:
    """A shallow copy of `message` with another content; the original is untouched."""
    clone = copy.copy(message)
    clone.extra = dict(message.extra)
    clone.content = content
    return clone


# ---------------------------------------------------------------------------
# Context and base class
# ---------------------------------------------------------------------------


@dataclass
class CompactionContext:
    """What a strategy needs to know; built by the buffer for each `compact()`."""

    count: Callable[[Message], int]
    """Local token estimate of one message."""

    tokenizer_fn: Callable[[Any], int]
    """Local token estimate of a text."""

    excess: int
    """Local tokens still to remove to reach the target. A strategy may stop
    as soon as it has removed that much; the orchestrator re-measures anyway."""

    budget: int
    """Local token budget of the whole history (the limit minus the system
    prompt, tool schemas and summary), for strategies that size their work on it."""

    summary: str = ""
    """Rolling summary of what was already folded (read and written by
    `SummarizeOlder`; the buffer stores it back)."""

    usage: AgentUsage = field(default_factory=AgentUsage)
    """Spend of the LLM calls made during this compaction."""

    preserve_first_user_message: bool = True
    preserve_last_user_message: bool = True

    notes: list[str] = field(default_factory=list)
    """Short human-readable notes on what happened (for telemetry / reports)."""

    def protected(self, messages: list[Message]) -> set[int]:
        return protected_indexes(messages, self.preserve_first_user_message, self.preserve_last_user_message)

    def total(self, messages: Iterable[Message]) -> int:
        return sum(self.count(m) for m in messages)


class CompactionStrategy:
    """Base class: `compact()` returns the same list when nothing changes."""

    name: str = "strategy"
    always: bool = False
    """True: run on every `compact()` even when the history is under the
    trigger (a lossless strategy such as `DeduplicateReads`)."""

    def compact(self, messages: list[Message], ctx: CompactionContext) -> list[Message]:  # pragma: no cover
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Tier 1: deduplicate file reads (lossless, always on)
# ---------------------------------------------------------------------------


@dataclass
class DeduplicateReads(CompactionStrategy):
    """Keep only the most recent read of a file; blank the older ones.

    The file identity comes from `file_key(name, arguments)`, which returns a
    key or None; the default knows the coding bundle (`ReadFileTool` and its
    `file_path`). With `identical_calls`, two calls with the same name and the
    same arguments are duplicates too, whatever the tool. Quasi lossless, so
    it runs on every request (`always`).
    """

    file_key: Callable[[str, Any], str | None] = default_file_key
    identical_calls: bool = True
    min_tokens: int = 50
    """Results smaller than this are left alone: not worth a cache miss."""
    placeholder: str = SUPERSEDED_READ
    always: bool = True
    name: str = "deduplicate_reads"

    def _key(self, name: str, arguments: Any) -> str | None:
        key = self.file_key(name, arguments) if self.file_key else None
        if key is None and self.identical_calls:
            key = "call:" + describe_call(name, arguments, max_chars=10_000)
        return key

    def compact(self, messages: list[Message], ctx: CompactionContext) -> list[Message]:
        calls = calls_by_id(messages)
        protected = ctx.protected(messages)
        seen: set[str] = set()
        result = list(messages)
        changed = False
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if message.role != "tool" or message.tool_call_id not in calls or index in protected:
                continue
            name, arguments = calls[message.tool_call_id]
            key = self._key(name, arguments)
            if key is None:
                continue
            if key in seen:
                if ctx.count(message) < self.min_tokens or self._is_placeholder(message):
                    continue
                if key.startswith("read:"):
                    text = self.placeholder.format(key=key[len("read:") :])
                else:
                    text = DUPLICATE_RESULT.format(call=describe_call(name, arguments))
                result[index] = with_content(message, text)
                changed = True
            seen.add(key)
        return result if changed else messages

    def _is_placeholder(self, message: Message) -> bool:
        return isinstance(message.content, str) and message.content.startswith("[")


# ---------------------------------------------------------------------------
# Tier 2: clamp oversized parts
# ---------------------------------------------------------------------------


@dataclass
class ClampOversized(CompactionStrategy):
    """Truncate one oversized part to a head and a tail with a marker.

    Applies to assistant text, tool results and, with `clamp_tool_call_args`,
    tool-call arguments: a model that wraps a whole file in its answer or in
    a tool argument must not fill the window on its own. User messages and
    pinned messages are never touched. The threshold is `max_part_tokens`,
    or `max_part_ratio` of the history budget when None.
    """

    max_part_tokens: int | None = None
    max_part_ratio: float = 0.25
    keep_head_chars: int = 2000
    keep_tail_chars: int = 2000
    clamp_tool_call_args: bool = True
    name: str = "clamp_oversized"

    def _threshold(self, ctx: CompactionContext) -> int:
        if self.max_part_tokens is not None:
            return max(int(self.max_part_tokens), 1)
        return max(int(ctx.budget * self.max_part_ratio), 1)

    def clamp_text(self, text: str) -> str:
        total = len(text)
        keep = self.keep_head_chars + self.keep_tail_chars
        if total <= keep:
            return text
        head = text[: self.keep_head_chars]
        tail = text[total - self.keep_tail_chars :] if self.keep_tail_chars else ""
        return head + CLAMP_MARKER.format(removed=total - keep, total=total) + tail

    def _clamp_content(self, content: Any, threshold: int, ctx: CompactionContext) -> tuple[Any, bool]:
        if isinstance(content, str):
            if ctx.tokenizer_fn(content) <= threshold:
                return content, False
            clamped = self.clamp_text(content)
            return clamped, clamped != content
        if isinstance(content, list):
            parts: list[Any] = []
            changed = False
            for part in content:
                if isinstance(part, TextContent) and ctx.tokenizer_fn(part.content) > threshold:
                    clamped = self.clamp_text(part.content)
                    if clamped != part.content:
                        part = TextContent(content=clamped)
                        changed = True
                parts.append(part)
            return (parts, True) if changed else (content, False)
        return content, False

    def _clamp_calls(self, calls: list[dict[str, Any]], threshold: int, ctx: CompactionContext) -> tuple[Any, bool]:
        new_calls: list[dict[str, Any]] = []
        changed = False
        for call in calls:
            function = call.get("function")
            arguments = function.get("arguments") if isinstance(function, dict) else call.get("arguments")
            if isinstance(arguments, str) and ctx.tokenizer_fn(arguments) > threshold:
                clamped = self.clamp_text(arguments)
                if clamped != arguments:
                    call = copy.deepcopy(call)
                    if isinstance(call.get("function"), dict):
                        call["function"]["arguments"] = clamped
                    else:
                        call["arguments"] = clamped
                    changed = True
            new_calls.append(call)
        return (new_calls, True) if changed else (calls, False)

    def compact(self, messages: list[Message], ctx: CompactionContext) -> list[Message]:
        threshold = self._threshold(ctx)
        protected = ctx.protected(messages)
        result = list(messages)
        changed = False
        for index, message in enumerate(messages):
            if message.role in ("user", "system") or index in protected:
                continue
            content, content_changed = self._clamp_content(message.content, threshold, ctx)
            calls, calls_changed = (
                self._clamp_calls(message.tool_calls, threshold, ctx)
                if self.clamp_tool_call_args and message.tool_calls
                else (message.tool_calls, False)
            )
            if content_changed or calls_changed:
                clone = with_content(message, content)
                clone.tool_calls = calls
                result[index] = clone
                changed = True
        return result if changed else messages


# ---------------------------------------------------------------------------
# Tier 3: clear old tool results
# ---------------------------------------------------------------------------


@dataclass
class ClearToolResults(CompactionStrategy):
    """Replace old tool results by a one-line stub, oldest first.

    The `keep_pairs` most recent call/result groups are kept whole, as are
    pinned results and the tools in `exclude_tools`. Results under
    `min_clear_tokens` are skipped: clearing them would invalidate the prompt
    cache for nothing. Stops once `ctx.excess` tokens are freed; with
    `clear_tool_inputs` the arguments of the cleared calls are blanked too.
    """

    keep_pairs: int = 3
    exclude_tools: frozenset[str] = frozenset()
    clear_tool_inputs: bool = False
    min_clear_tokens: int | None = 100
    placeholder: str = CLEARED_RESULT
    name: str = "clear_tool_results"

    def compact(self, messages: list[Message], ctx: CompactionContext) -> list[Message]:
        calls = calls_by_id(messages)
        protected = ctx.protected(messages)
        groups = [(start, end) for start, end in tool_groups(messages) if end - start > 1]
        candidates = groups[: max(len(groups) - self.keep_pairs, 0)]
        result = list(messages)
        freed = 0
        changed = False
        for start, end in candidates:
            if freed >= ctx.excess:
                break
            for index in range(start + 1, end):
                message = messages[index]
                if index in protected or message.tool_call_id not in calls:
                    continue
                name, arguments = calls[message.tool_call_id]
                if name in self.exclude_tools or self._is_stub(message):
                    continue
                tokens = ctx.count(message)
                if self.min_clear_tokens is not None and tokens < self.min_clear_tokens:
                    continue
                stub = with_content(
                    message, self.placeholder.format(call=describe_call(name, arguments), tokens=tokens)
                )
                result[index] = stub
                freed += tokens - ctx.count(stub)
                changed = True
            if self.clear_tool_inputs and start not in protected:
                assistant = result[start]
                blanked = self._blank_arguments(assistant)
                if blanked is not assistant:
                    freed += ctx.count(assistant) - ctx.count(blanked)
                    result[start] = blanked
                    changed = True
        return result if changed else messages

    @staticmethod
    def _is_stub(message: Message) -> bool:
        return isinstance(message.content, str) and message.content.startswith("[")

    @staticmethod
    def _blank_arguments(message: Message) -> Message:
        calls = []
        changed = False
        for call in message.tool_calls or []:
            function = call.get("function")
            current = function.get("arguments") if isinstance(function, dict) else call.get("arguments")
            if current in (None, CLEARED_ARGUMENTS):
                calls.append(call)
                continue
            call = copy.deepcopy(call)
            if isinstance(call.get("function"), dict):
                call["function"]["arguments"] = CLEARED_ARGUMENTS
            else:
                call["arguments"] = CLEARED_ARGUMENTS
            calls.append(call)
            changed = True
        if not changed:
            return message
        clone = with_content(message, message.content)
        clone.tool_calls = calls
        return clone


# ---------------------------------------------------------------------------
# Tier 4: summarize older messages
# ---------------------------------------------------------------------------

SUMMARY_PROMPT = (
    "You maintain a running summary of an ongoing conversation between a user and an "
    "AI agent. Merge the previous summary (if any) with the new messages into a single, "
    "dense summary. Preserve: the user's goals and constraints, decisions made, key facts "
    "and figures, tool results that still matter, and any unresolved questions. Drop "
    "greetings and redundancy. Answer with the summary only, no preamble."
)


@dataclass
class SummarizeOlder(CompactionStrategy):
    """Fold the oldest messages into the rolling summary; keep a recent tail.

    Works inside the live turn too: a long tool loop ends up with the task
    (protected), the summary of its first steps, and a recent tail verbatim.
    The tail is the most recent groups that fit in `keep_tokens` tokens
    (`keep_ratio` of the history budget when None) and at most `keep_messages`
    messages; the cut is a group boundary. Protected messages before the cut
    are kept in place, not summarized.

    `llm` is any object with `invoke(input=list[dict]) -> response` whose
    response has `.content` (and optionally `.usage`). An empty summary is
    retried once, then the strategy gives up for this compaction and the
    orchestrator moves on to the next tier.
    """

    llm: Any = None
    keep_messages: int | None = 20
    keep_tokens: int | None = None
    keep_ratio: float = 0.5
    max_summary_ratio: float = 0.15
    """Cap of the summary, as a share of the history budget."""
    prompt: str = SUMMARY_PROMPT
    tool_result_max_chars: int | None = 500
    retries: int = 1
    name: str = "summarize_older"

    def _cut(self, messages: list[Message], ctx: CompactionContext) -> int:
        """Index of the first message kept verbatim (the tail), on a group boundary."""
        tail_budget = self.keep_tokens if self.keep_tokens is not None else int(ctx.budget * self.keep_ratio)
        kept_tokens = 0
        kept_count = 0
        cut = len(messages)
        for start, end in reversed(tool_groups(messages)):
            size = ctx.total(messages[start:end])
            if kept_tokens + size > tail_budget:
                break
            if self.keep_messages is not None and kept_count + (end - start) > self.keep_messages:
                break
            kept_tokens += size
            kept_count += end - start
            cut = start
        return cut

    def render(self, messages: list[Message]) -> str:
        from gwenflow.memory.base import render_messages_as_text

        return render_messages_as_text(messages, max_chars_per_message=self.tool_result_max_chars)

    def summarize(self, previous: str, messages: list[Message], ctx: CompactionContext) -> str | None:
        text = self.render(messages)
        if not text:
            return previous or ""
        user_prompt = f"Previous summary:\n{previous}\n\n" if previous else ""
        user_prompt += f"New messages to fold in:\n{text}"
        request = [{"role": "system", "content": self.prompt}, {"role": "user", "content": user_prompt}]
        for _attempt in range(self.retries + 1):
            response = self.llm.invoke(input=request)
            usage = getattr(response, "usage", None)
            if usage is not None:
                ctx.usage.add(usage)
            summary = (getattr(response, "content", None) or "").strip()
            if summary:
                return summary
            logger.warning("Summarizer returned an empty summary, retrying." if _attempt < self.retries else "")
        return None

    def compact(self, messages: list[Message], ctx: CompactionContext) -> list[Message]:
        if self.llm is None:
            return messages
        cut = self._cut(messages, ctx)
        protected = ctx.protected(messages)
        to_fold = [m for i, m in enumerate(messages[:cut]) if i not in protected]
        if not to_fold:
            return messages
        summary = self.summarize(ctx.summary, to_fold, ctx)
        if summary is None:
            logger.warning("History summary failed (empty answer): falling back to the next compaction tier.")
            return messages
        from gwenflow.utils.tokens import keep_tokens_from_text

        max_tokens = max(int(ctx.budget * self.max_summary_ratio), 1)
        ctx.summary = keep_tokens_from_text(summary, token_limit=max_tokens, tokenizer_fn=ctx.tokenizer_fn)
        ctx.notes.append(f"summarized {len(to_fold)} message(s)")
        kept_before_cut = [m for i, m in enumerate(messages[:cut]) if i in protected]
        return kept_before_cut + list(messages[cut:])


# ---------------------------------------------------------------------------
# Tier 5: sliding window (last resort)
# ---------------------------------------------------------------------------


@dataclass
class SlidingWindow(CompactionStrategy):
    """Drop the oldest groups until `ctx.excess` tokens are gone.

    Protected messages (first user message, pinned) are kept in place; a
    group is dropped whole. Lossy: the last resort when nothing else fits.
    """

    keep_messages: int | None = None
    """When set, drop until at most this many messages remain (besides the
    protected ones), whatever the token excess."""
    name: str = "sliding_window"

    def compact(self, messages: list[Message], ctx: CompactionContext) -> list[Message]:
        protected = ctx.protected(messages)
        drop: set[int] = set()
        freed = 0
        remaining = len(messages)
        for start, end in tool_groups(messages):
            over_count = self.keep_messages is not None and remaining > self.keep_messages
            if freed >= ctx.excess and not over_count:
                break
            if any(i in protected for i in range(start, end)):
                continue
            size = ctx.total(messages[start:end])
            drop.update(range(start, end))
            freed += size
            remaining -= end - start
        if not drop:
            return messages
        ctx.notes.append(f"dropped {len(drop)} message(s)")
        return [m for i, m in enumerate(messages) if i not in drop]


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass
class CompactionRecord:
    """What one `compact()` did: for telemetry, reports and tests."""

    strategies: list[str]
    messages_before: int
    messages_after: int
    tokens_before: int
    tokens_after: int
    notes: list[str] = field(default_factory=list)
    removed: list[Message] = field(default_factory=list)
    """The original messages that were altered or dropped, for inspection."""


def default_strategies(summarizer: Any = None, **overrides: Any) -> list[CompactionStrategy]:
    """The default tiers, cheapest first; `summarizer=None` skips the summary tier."""
    tiers: list[CompactionStrategy] = [
        DeduplicateReads(**overrides.get("deduplicate", {})),
        ClampOversized(**overrides.get("clamp", {})),
        ClearToolResults(**overrides.get("clear", {})),
    ]
    if summarizer is not None:
        tiers.append(SummarizeOlder(llm=summarizer, **overrides.get("summarize", {})))
    tiers.append(SlidingWindow(**overrides.get("sliding", {})))
    return tiers
