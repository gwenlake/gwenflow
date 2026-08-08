import asyncio
from dataclasses import dataclass, field
from typing import Any

from gwenflow.logger import logger
from gwenflow.memory.chat_memory_buffer import ChatMemoryBuffer
from gwenflow.types import AgentUsage, Message, TextContent
from gwenflow.utils.tokens import keep_tokens_from_text

MAX_SUMMARY_CONTENT = 0.15

SUMMARY_PROMPT = (
    "You maintain a running summary of an ongoing conversation between a user and an "
    "AI agent. Merge the previous summary (if any) with the new messages into a single, "
    "dense summary. Preserve: the user's goals and constraints, decisions made, key facts "
    "and figures, tool results that still matter, and any unresolved questions. Drop "
    "greetings and redundancy. Answer with the summary only, no preamble."
)


@dataclass
class ChatSummaryMemoryBuffer(ChatMemoryBuffer):
    """ChatMemoryBuffer that compacts evicted messages instead of dropping them.

    Compaction is an explicit command: `compact()` (or `await acompact()`)
    simulates the prune, folds the messages that would be evicted into a rolling
    summary (`llm` calls), and stops there. `get()` stays a pure read that
    simply sends the current summary as the first message of the window — its
    token cost is deducted from the budget by `_budget()`, and it is capped at
    ``MAX_SUMMARY_CONTENT * token_limit`` tokens.

    The agent calls `compact()` / `acompact()` before each `get()`.

    Guarantees:
      - The live turn is never summarized.
      - Each message is folded into the summary at most once (tracked by
        ``_summarized_upto``).
      - `compact()` iterates to a fixed point: since the summary itself takes
        budget, folding can evict a few more messages — those are folded in the
        same call, so the next `get()` never silently drops an unsummarized
        message (other than inside the live turn).
      - If the LLM call fails, nothing is lost: the pointer does not advance
        and the same messages are retried on the next `compact()`. Meanwhile
        the buffer degrades to the plain drop behaviour of ``ChatMemoryBuffer``.
    """

    llm: Any = None
    """LLM used to compress evicted messages (any object exposing
    `.invoke(input=list[dict]) -> response` with a `.content` attribute,
    e.g. a gwenflow ChatBase). Give it a plain instance — no tools, no
    response_format. None disables compaction."""

    summary: str = ""
    """Rolling summary of the messages already evicted from the window."""

    _summarized_upto: int = 0
    """Index into `self.messages`: everything before it is folded in `summary`."""

    usage: AgentUsage = field(default_factory=AgentUsage)
    """Cumulative usage of the compaction LLM calls. Deliberately NOT cleared
    by `reset()`: tokens already spent stay accounted for (see /cost)."""

    def reset(self):
        super().reset()
        self.summary = ""
        self._summarized_upto = 0

    # ------------------------------------------------------------------
    # ChatMemoryBuffer hooks and commands
    # ------------------------------------------------------------------

    def _summary_message(self) -> Message | None:
        if not self.summary:
            return None
        return Message(
            role="assistant",
            content=f"Summary of the earlier conversation (older messages were compressed):\n\n{self.summary}",
        )

    def compact(self) -> None:
        """Fold the messages that the next `get()` would evict into the summary.

        Iterates until no more messages are evictable (the summary takes
        budget, so one fold can trigger the next). Incremental: a message is
        folded at most once. On LLM failure, stops without advancing so the
        next `compact()` retries.
        """
        if self.llm is None or not self.messages:
            return

        while True:
            budget = self._budget()
            chat_history = self._clamp_history(list(self.messages), budget)
            start, anchor, _ = self._prune(chat_history, budget)

            # Never summarize the live turn: stop at the anchor.
            evict_end = start if anchor is None else min(start, anchor)
            if evict_end <= self._summarized_upto:
                return
            if not self._compress_evicted(self.messages[self._summarized_upto : evict_end]):
                return  # LLM failed: keep the pointer, retry on a later compact()
            self._summarized_upto = evict_end

    async def acompact(self) -> None:
        """Async variant of `compact()`.

        Runs the (sync) LLM calls in a worker thread so the event loop is
        never blocked.
        """
        await asyncio.to_thread(self.compact)

    # ------------------------------------------------------------------
    # Compression
    # ------------------------------------------------------------------

    def _render_for_summary(self, messages: list[Message]) -> str:
        """Plain-text rendering of evicted messages for the summarizer."""
        lines: list[str] = []
        for m in messages:
            if isinstance(m.content, str) and m.content:
                lines.append(f"[{m.role}] {m.content}")
            elif isinstance(m.content, list):
                texts = [p.content for p in m.content if isinstance(p, TextContent)]
                if texts:
                    lines.append(f"[{m.role}] " + " ".join(texts))
            if m.tool_calls:
                for tc in m.tool_calls:
                    lines.append(f"[{m.role}] tool_call: {tc}")
        return "\n".join(lines)

    def _compress_evicted(self, evicted: list[Message]) -> bool:
        """Fold `evicted` into the rolling summary.

        Returns True when the messages are accounted for (folded, or nothing to
        fold) and the pointer may advance; False on LLM failure so the caller
        keeps the pointer and retries later.
        """
        text = self._render_for_summary(evicted)
        if not text:
            return True  # nothing textual to fold: safe to advance
        user_prompt = ""
        if self.summary:
            user_prompt += f"Previous summary:\n{self.summary}\n\n"
        user_prompt += f"New messages to fold in:\n{text}"
        try:
            response = self.llm.invoke(
                input=[
                    {"role": "system", "content": SUMMARY_PROMPT},
                    {"role": "user", "content": user_prompt},
                ]
            )
            new_summary = (getattr(response, "content", None) or "").strip()
        except Exception as e:  # noqa: BLE001 — compaction is best-effort
            logger.warning(f"History compaction failed, will retry on next compact(): {e}")
            return False
        response_usage = getattr(response, "usage", None)
        if response_usage is not None:
            self.usage.add(response_usage)
        if not new_summary:
            logger.warning("History compaction returned an empty summary, will retry on next compact().")
            return False
        max_summary_tokens = int(MAX_SUMMARY_CONTENT * self.token_limit)
        self.summary = keep_tokens_from_text(
            new_summary, token_limit=max_summary_tokens, tokenizer_fn=self.tokenizer_fn
        )
        return True
