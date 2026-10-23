from dataclasses import dataclass


@dataclass(frozen=True)
class ContextWindowInfo:
    """Context window of a model and how it was found.

    `resolved` is False when the value is assumed (`fallback_context_window`
    or the library default) rather than known (explicit, discovered from the
    server, or listed in `MODELS`).
    """

    tokens: int
    resolved: bool = True
    source: str = "table"


@dataclass(frozen=True)
class ContextUsage:
    """Size of the next request, as the memory buffer sees it.

    `used_tokens` is anchored on the provider's last reported usage when one
    is available (see `ChatMemoryBuffer.observe_usage`), otherwise estimated
    locally. `window_tokens` is the model's context window, `limit_tokens`
    the budget the buffer compacts to, and `resolved` whether the window is
    known or assumed.
    """

    used_tokens: int
    window_tokens: int
    limit_tokens: int
    resolved: bool = True
    anchored: bool = False

    @property
    def fraction(self) -> float:
        """Share of the context window the next request would use."""
        return self.used_tokens / self.window_tokens if self.window_tokens else 0.0
