from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from gwenflow.tools import BaseTool

LLM_MODEL_PARAMETERS = {
    # --- OPENAI ---
    "gpt-5.2": {
        "context_token": 400000,
        "reasoning": True
    },
    "gpt-5.2-pro": {
        "context_token": 400000,
        "reasoning": True
    },
    "gpt-5-mini": {
        "context_token": 400000,
        "reasoning": True
    },
    "gpt-4.5": {
        "context_token": 128000,
        "reasoning": False
    },
    "gpt-4.1": {
        "context_token": 1047576,
        "reasoning": False
    },
    "o3-pro": {
        "context_token": 200000,
        "reasoning": True
    },
    "o3-mini-high": {
        "context_token": 128000,
        "reasoning": True
    },
    "o4-mini": {
        "context_token": 128000,
        "reasoning": True
    },
    "gpt-4o": {
        "context_token": 128000,
        "reasoning": False
    },
    "gpt-4o-mini": {
        "context_token": 128000,
        "reasoning": False
    },
    "o1-preview": {
        "context_token": 128000,
        "reasoning": True
    },
    "o1-mini": {
        "context_token": 128000,
        "reasoning": True
    },

    # --- DEEPSEEK ---
    "deepseek-chat": {
        "context_token": 128000,
        "reasoning": False
    },
    "deepseek-r1": {
        "context_token": 128000,
        "reasoning": True
    },

    # --- META ---
    "llama-3.1-70b-versatile": {
        "context_token": 131072,
        "reasoning": False
    },
    "llama-3.1-8b-instant": {
        "context_token": 131072,
        "reasoning": False
    },

    # --- MISTRAL ---
    "mixtral-8x7b-32768": {
        "context_token": 32768,
        "reasoning": False
    },
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
    def invoke(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    async def ainvoke(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def stream(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    async def astream(self, *args, **kwargs) -> Any:
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
