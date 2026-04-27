from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any, Dict, Iterator, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from gwenflow.tools import BaseTool
from gwenflow.types import ModelResponse

LLM_MODEL_PARAMETERS = {
    # --- OPENAI ---
    "gpt-5.4-pro": {"context_token": 1050000, "reasoning": True},
    "gpt-5.4": {"context_token": 1050000, "reasoning": True},
    "gpt-5.4-mini": {"context_token": 400000, "reasoning": True},
    "gpt-5.4-nano": {"context_token": 400000, "reasoning": True},
    "gpt-5.2": {"context_token": 400000, "reasoning": True},
    "gpt-5.2-pro": {"context_token": 400000, "reasoning": True},
    "gpt-5": {"context_token": 400000, "reasoning": True},
    "gpt-5-mini": {"context_token": 400000, "reasoning": True},
    "gpt-5-nano": {"context_token": 400000, "reasoning": True},
    "gpt-4.5": {"context_token": 128000, "reasoning": False},
    "gpt-4.1": {"context_token": 1047576, "reasoning": False},
    "o3-pro": {"context_token": 200000, "reasoning": True},
    "o3-mini-high": {"context_token": 128000, "reasoning": True},
    "o4-mini": {"context_token": 128000, "reasoning": True},
    "gpt-4o": {"context_token": 128000, "reasoning": False},
    "gpt-4o-mini": {"context_token": 128000, "reasoning": False},
    "o1-preview": {"context_token": 128000, "reasoning": True},
    "o1-mini": {"context_token": 128000, "reasoning": True},
    # --- ANTHROPIC ---
    "claude-opus-4-7": {"context_token": 1000000, "reasoning": True},
    "claude-opus-4-6": {"context_token": 1000000, "reasoning": True},
    "claude-opus-4-5": {"context_token": 200000, "reasoning": True},
    "claude-opus-4-1": {"context_token": 200000, "reasoning": True},
    "claude-sonnet-4-6": {"context_token": 1000000, "reasoning": False},
    "claude-sonnet-4-5": {"context_token": 200000, "reasoning": False},
    "claude-haiku-4-5": {"context_token": 200000, "reasoning": False},
    # --- GOOGLE ---
    "gemini-3.1-pro-preview": {"context_token": 1048576, "reasoning": True},
    "gemini-3.1-flash-lite-preview": {"context_token": 1048576, "reasoning": True},
    "gemini-3-flash-preview": {"context_token": 1048576, "reasoning": True},
    "gemini-2.5-pro": {"context_token": 1048576, "reasoning": True},
    "gemini-2.5-flash": {"context_token": 1048576, "reasoning": True},
    "gemini-2.5-flash-lite": {"context_token": 1048576, "reasoning": True},
    "gemini-2.0-pro": {"context_token": 2097152, "reasoning": False},
    "gemini-2.0-flash": {"context_token": 1048576, "reasoning": False},
    "gemini-2.0-flash-thinking": {"context_token": 1048576, "reasoning": True},
    "gemini-1.5-pro": {"context_token": 2097152, "reasoning": False},
    "gemini-1.5-flash": {"context_token": 1048576, "reasoning": False},
    # --- DEEPSEEK ---
    "deepseek-chat": {"context_token": 128000, "reasoning": False},
    "deepseek-r1": {"context_token": 128000, "reasoning": True},
    # --- META ---
    "llama-3.1-70b-versatile": {"context_token": 131072, "reasoning": False},
    "llama-3.1-8b-instant": {"context_token": 131072, "reasoning": False},
    # --- MISTRAL ---
    "mistral-large-2512": {"context_token": 256000, "reasoning": False},
    "mistral-small-2603": {"context_token": 256000, "reasoning": False},
    "mistral-small-2506": {"context_token": 128000, "reasoning": False},
    "ministral-14b-2512": {"context_token": 256000, "reasoning": False},
    "ministral-8b-2512": {"context_token": 256000, "reasoning": False},
    "ministral-3b-2512": {"context_token": 256000, "reasoning": False},
    "mixtral-8x7b-32768": {"context_token": 32768, "reasoning": False},
}


class ChatBase(BaseModel, ABC):
    model: str
    """The model to use when invoking the LLM."""

    system_prompt: Optional[str] = None
    """The system prompt to use when invoking the LLM."""

    tools: List[BaseTool] = Field(default_factory=list)
    """A list of tools that the LLM can use."""

    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    tool_type: str = Field(default="fncall")

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

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

    def get_context_window_size(self) -> int:
        # Only using 75% of the context window size to avoid cutting the message in the middle
        return int(LLM_MODEL_PARAMETERS.get(self.model, {}).get("context_token", 128000) * 0.75)

    def get_reasoning_model(self) -> str:
        return LLM_MODEL_PARAMETERS.get(self.model, {}).get("reasoning", False)

    def get_tool_names(self):
        return [tool.name for tool in self.tools]

    def get_tool_map(self):
        return {tool.name: tool for tool in self.tools}
