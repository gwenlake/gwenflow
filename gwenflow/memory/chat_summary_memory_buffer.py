from dataclasses import dataclass
from typing import Any

from gwenflow.memory.chat_memory_buffer import ChatMemoryBuffer
from gwenflow.memory.compaction import SUMMARY_PROMPT, CompactionRecord, CompactionStrategy, default_strategies
from gwenflow.types import Message

__all__ = ["ChatSummaryMemoryBuffer", "SUMMARY_PROMPT", "MAX_SUMMARY_CONTENT"]

MAX_SUMMARY_CONTENT = 0.15


@dataclass
class ChatSummaryMemoryBuffer(ChatMemoryBuffer):
    """ChatMemoryBuffer whose compaction can summarize instead of dropping.

    Same tiers as the base buffer, with `SummarizeOlder` (on `llm`) inserted
    before the sliding window: once deduplicating, clamping and clearing tool
    results are not enough, the oldest messages are folded into a rolling
    `summary`, which `get()` sends as the first message of the window. The
    summary's cost is deducted from the budget and capped at
    ``MAX_SUMMARY_CONTENT`` of it.

    The task of the run (pinned by the agent) and the first user message are
    kept verbatim; everything older than the summary tier's tail can be
    summarized, inside the live turn included. A summarizer that fails or
    returns nothing never fails the run: the sliding window takes over for
    that compaction and the summary is retried on the next one.
    """

    llm: Any = None
    """LLM used to summarize (any object exposing `.invoke(input=list[dict])
    -> response` with a `.content` attribute, e.g. a gwenflow ChatBase).
    Give it a plain instance — no tools, no response_format. None disables
    the summary tier (the buffer then behaves like `ChatMemoryBuffer`)."""

    summary: str = ""
    """Rolling summary of the messages already folded."""

    _summarized_upto: int = 0
    """Number of messages folded into the summary so far (reports, /save)."""

    def reset(self):
        super().reset()
        self.summary = ""
        self._summarized_upto = 0

    # ------------------------------------------------------------------
    # ChatMemoryBuffer hooks
    # ------------------------------------------------------------------

    def _summarizer(self) -> Any:
        return self.llm

    def _default_strategies(self) -> list[CompactionStrategy]:
        return default_strategies(summarizer=self.llm, summarize={"max_summary_ratio": MAX_SUMMARY_CONTENT})

    def _store_summary(self, summary: str) -> None:
        self.summary = summary

    def _summary_message(self) -> Message | None:
        if not self.summary:
            return None
        return Message(
            role="assistant",
            content=(
                "Summary of the earlier conversation (older messages were compressed; "
                "what follows is second-hand, verify before relying on details):\n\n"
                f"{self.summary}"
            ),
        )

    def compact(self) -> CompactionRecord | None:
        before = len(self.messages)
        record = super().compact()
        if record is not None and "summarize_older" in record.strategies:
            self._summarized_upto += before - len(self.messages)
        return record
