from __future__ import annotations as _annotations

from dataclasses import dataclass, field


@dataclass(kw_only=True)
class RequestUsage:
    input_tokens: int = 0
    """Number of input tokens."""
    cache_write_tokens: int = 0
    """Number of tokens written to the cache."""
    cache_read_tokens: int = 0
    """Number of tokens read from the cache."""
    output_tokens: int = 0

    input_audio_tokens: int = 0
    """Number of audio input tokens."""
    cache_audio_read_tokens: int = 0
    """Number of audio tokens read from the cache."""
    output_audio_tokens: int = 0
    """Number of audio output tokens."""

    details: dict[str, int] = field(default_factory=dict[str, int])

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(kw_only=True)
class AgentUsage(RequestUsage):
    requests: int = 0
    """Number of requests made to the LLM API."""

    tool_calls: int = 0
    """Number of successful tool calls executed during the run."""

    def add(self, other: RequestUsage | AgentUsage) -> None:
        if isinstance(other, AgentUsage):
            self.requests += other.requests
            self.tool_calls += other.tool_calls
        else:
            self.requests += 1

        self.input_tokens += other.input_tokens
        self.cache_write_tokens += other.cache_write_tokens
        self.cache_read_tokens += other.cache_read_tokens
        self.input_audio_tokens += other.input_audio_tokens
        self.cache_audio_read_tokens += other.cache_audio_read_tokens
        self.output_tokens += other.output_tokens

        for key, value in other.details.items():
            if isinstance(value, (int, float)):
                self.details[key] = self.details.get(key, 0) + value
