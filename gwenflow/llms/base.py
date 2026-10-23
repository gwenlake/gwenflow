import copy
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Union

from gwenflow.llms.models import MODELS, ModelInfo
from gwenflow.logger import logger
from gwenflow.tools import BaseTool
from gwenflow.types import Message, ModelResponse, ThinkingContent
from gwenflow.types.context import ContextWindowInfo

DEFAULT_CONTEXT_SIZE = 128000

CONTEXT_SAFETY_RATIO = 0.75
"""Share of the context window the agent lets its history use; the rest is left
for the model's answer and for estimation errors."""

_MODEL_NAME_SEPARATORS = "-_:@"
_warned_unknown_models: set[str] = set()

# Case-insensitive view of the table, built once: every comparison below is
# done on lowercase names, whatever the casing of the key or the request.
_MODELS_LOWER: dict[str, ModelInfo] = {key.lower(): info for key, info in MODELS.items()}


def normalize_model_name(model: str) -> str:
    """Lowercase a model id and strip any provider / organisation prefix.

    `Qwen/Qwen3.6-35B-A3B` -> `qwen3.6-35b-a3b`, `GPT-5-Mini` -> `gpt-5-mini`.
    """
    return model.strip().lower().rsplit("/", 1)[-1]


def lookup_model_info(model: str) -> Optional[ModelInfo]:
    """Find a model in `MODELS`, tolerating the names inference servers use.

    Matching ignores case throughout. Tries, in order: the name lowercased and
    stripped of any provider / organisation prefix (`Qwen/Qwen3.6-35B-A3B` ->
    `qwen3.6-35b-a3b`); then the longest known name the model id starts with,
    followed by a separator (`mistral-small-3.2-24b-instruct-2506` ->
    `mistral-small-3.2-24b`). A dot is not a separator, so `gpt-5.7` never
    falls back on `gpt-5`.
    """
    if not model:
        return None
    name = normalize_model_name(model)
    if name in _MODELS_LOWER:
        return _MODELS_LOWER[name]
    matches = [
        key
        for key in _MODELS_LOWER
        if name.startswith(key) and len(name) > len(key) and name[len(key)] in _MODEL_NAME_SEPARATORS
    ]
    if matches:
        return _MODELS_LOWER[max(matches, key=len)]
    return None


@dataclass(kw_only=True)
class ChatBase(ABC):
    model: str
    """The model to use when invoking the LLM."""

    system_prompt: Optional[str] = None
    """The system prompt to use when invoking the LLM."""

    tools: List[BaseTool] = field(default_factory=list)
    """A list of tools that the LLM can use."""

    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    tool_type: str = "fncall"

    provider: Optional[str] = None

    context_window: Optional[Union[int, str]] = None
    """Context window of the model, in tokens. None looks the model up in
    `MODELS`; an int overrides it (a vLLM server started with a smaller
    `--max-model-len`, a model missing from the table); "auto" asks the
    server (see `discover_context_window`) and falls back on the table."""

    fallback_context_window: Optional[int] = None
    """Context window to assume when the model is unknown: not in `MODELS`,
    not discovered. Unlike `context_window` it never overrides a known value.
    None falls back on `DEFAULT_CONTEXT_SIZE` with a warning."""

    def _forced_tool_name(self) -> Optional[str]:
        """Name of the tool a named `tool_choice` forces, or None.

        The agent forces its `final_result` tool with the OpenAI shape
        `{"type": "function", "function": {"name": ...}}`; each provider
        translates it into its own dialect from here.
        """
        choice = self.tool_choice
        if isinstance(choice, dict) and choice.get("type") == "function":
            return (choice.get("function") or {}).get("name") or choice.get("name")
        return None

    @abstractmethod
    def invoke(self, *args, **kwargs) -> ModelResponse:
        pass

    @abstractmethod
    async def ainvoke(self, *args, **kwargs) -> ModelResponse:
        pass

    @abstractmethod
    def stream(self, *args, **kwargs) -> Iterator[ModelResponse]:
        pass

    @abstractmethod
    async def astream(self, *args, **kwargs) -> AsyncIterator[ModelResponse]:
        pass

    async def aclose(self) -> None:
        async_client = getattr(self, "async_client", None)
        if async_client is not None:
            await async_client.close()
            self.async_client = None

    def discover_context_window(self) -> Optional[int]:
        """Ask the provider for the model's context window. None if unsupported."""
        return None

    def context_window_info(self) -> ContextWindowInfo:
        """Context window of the model and whether it is known or assumed.

        Resolution order: an explicit `context_window` (int); `"auto"`
        discovery from the server; the `MODELS` table; `fallback_context_window`;
        `DEFAULT_CONTEXT_SIZE` with a one-time warning. Only the last two are
        reported as not resolved.
        """
        if isinstance(self.context_window, int) and self.context_window > 0:
            return ContextWindowInfo(tokens=self.context_window, resolved=True, source="context_window")
        if self.context_window == "auto":
            discovered = self.discover_context_window()
            if discovered:
                self.context_window = discovered  # cache: ask the server once
                return ContextWindowInfo(tokens=discovered, resolved=True, source="server")
        info = lookup_model_info(self.model)
        if info is not None:
            return ContextWindowInfo(tokens=info["context_window"], resolved=True, source="table")
        if self.fallback_context_window:
            return ContextWindowInfo(tokens=self.fallback_context_window, resolved=False, source="fallback")
        if self.model not in _warned_unknown_models:
            _warned_unknown_models.add(self.model)
            logger.warning(
                f"Unknown context window for model '{self.model}': assuming {DEFAULT_CONTEXT_SIZE:,} tokens. "
                f"Pass context_window=<tokens> (or 'auto' for a vLLM-compatible server), or "
                f"fallback_context_window=<tokens>, if that is wrong."
            )
        return ContextWindowInfo(tokens=DEFAULT_CONTEXT_SIZE, resolved=False, source="default")

    def resolve_context_window(self) -> int:
        """Context window of the model, in tokens (see `context_window_info`)."""
        return self.context_window_info().tokens

    def get_context_size(self) -> int:
        """Token budget for the agent's history: the window minus a safety margin."""
        return int(self.resolve_context_window() * CONTEXT_SAFETY_RATIO)

    def get_reasoning_model(self) -> str:
        info = lookup_model_info(self.model)
        return info["reasoning"] if info is not None else False

    def get_thinking_parts(self, response: ModelResponse) -> Optional[List[ThinkingContent]]:
        """Thinking parts to attach to the assistant Message for echo-back.

        Default: None. Only Anthropic extended thinking requires this — signatures
        (carried in ThinkingContent.extra) must be replayed verbatim on subsequent
        turns when tool use is involved.
        """
        return None

    def get_tool_names(self):
        return [tool.name for tool in self.tools]

    def get_tool_map(self):
        return {tool.name: tool for tool in self.tools}

    def input_to_message_list(
        self,
        input: Union[str, List[Message], List[Dict[str, str]]],
    ) -> List[Message]:
        """Converts a string or list of messages into a list of messages."""
        if isinstance(input, str):
            return [Message(role="user", content=input)]
        messages = copy.deepcopy(input)
        for i, message in enumerate(messages):
            if isinstance(message, dict):
                messages[i] = Message(**message)
        return messages
