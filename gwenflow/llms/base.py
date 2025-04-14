from typing import Optional, Union, Any, List, Dict
from pydantic import BaseModel, ConfigDict, Field
from abc import ABC, abstractmethod

from gwenflow.logger import logger
from gwenflow.tools import BaseTool
from gwenflow.types import Message


LLM_CONTEXT_WINDOW_SIZES = {
    # openai
    "gpt-4": 8192,
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-4-turbo": 128000,
    "o1-preview": 128000,
    "o1-mini": 128000,
    "o3-mini": 128000,
    # deepseek
    "deepseek-chat": 128000,
    "deepseek-r1": 128000,
    # google
    "gemma2-9b-it": 8192,
    "gemma-7b-it": 8192,
    # meta
    "llama3-groq-70b-8192-tool-use-preview": 8192,
    "llama3-groq-8b-8192-tool-use-preview": 8192,
    "llama-3.1-70b-versatile": 131072,
    "llama-3.1-8b-instant": 131072,
    "llama-3.2-1b-preview": 8192,
    "llama-3.2-3b-preview": 8192,
    "llama-3.2-11b-text-preview": 8192,
    "llama-3.2-90b-text-preview": 8192,
    "llama3-70b-8192": 8192,
    "llama3-8b-8192": 8192,
    # mistral
    "mixtral-8x7b-32768": 32768,
}

class ChatBase(BaseModel, ABC):
 
    model: str
    """The model to use when invoking the LLM."""

    system_prompt: Optional[str] = None
    """The system prompt to use when invoking the LLM."""

    tools: List[BaseTool] = Field(default_factory=list)
    """A list of tools that the LLM can use."""

    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    
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
        return int(LLM_CONTEXT_WINDOW_SIZES.get(self.model, 8192) * 0.75)

    def get_tool_names(self):
        return [tool.name for tool in self.tools]

    def get_tool_map(self):
        return {tool.name: tool for tool in self.tools}
