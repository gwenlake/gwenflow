import copy
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Union

from gwenflow.llms.models import MODELS
from gwenflow.tools import Tool
from gwenflow.types import Message, ModelResponse

DEFAULT_CONTEXT_SIZE = 128000


@dataclass(kw_only=True)
class ChatBase(ABC):
    model: str
    """The model to use when invoking the LLM."""

    system_prompt: Optional[str] = None
    """The system prompt to use when invoking the LLM."""

    tools: List[Tool] = field(default_factory=list)
    """A list of tools that the LLM can use."""

    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    tool_type: str = "fncall"

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
        """Close the async client (if any) so its httpx connection pool is shut down
        on the current event loop. Prevents 'Event loop is closed' warnings at teardown."""
        async_client = getattr(self, "async_client", None)
        if async_client is not None:
            await async_client.close()
            self.async_client = None

    def get_context_size(self) -> int:
        return int(MODELS.get(self.model, {}).get("context_window", DEFAULT_CONTEXT_SIZE) * 0.75)

    def get_reasoning_model(self) -> str:
        return MODELS.get(self.model, {}).get("reasoning", False)

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
