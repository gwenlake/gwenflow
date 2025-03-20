from typing import Optional, Union, Any, List, Dict
from pydantic import BaseModel, ConfigDict
from abc import ABC, abstractmethod

import asyncio
import json

from gwenflow.tools import BaseTool
from gwenflow.types import Message, ChatCompletionMessageToolCall
from gwenflow.utils import logger


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
    system_prompt: Optional[str] = None
    tools: List[BaseTool] = []
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

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

    def _cast_messages(self, messages: Union[str, List[Message], List[Dict[str, str]]],) -> List[Message]:
        if isinstance(messages, str):
            _messages = [Message(role="user", content=messages)]
        else:
            _messages = messages
            for i, message in enumerate(_messages):
                if not isinstance(message, Message):
                    _messages[i] = Message(**message)
        return _messages

    def get_tool_names(self):
        return [tool.name for tool in self.tools]

    def get_tools_map(self):
        return {tool.name: tool for tool in self.tools}

    def handle_tool_call(self, tool_call) -> Message:

        if isinstance(tool_call, dict):
            tool_call = ChatCompletionMessageToolCall(**tool_call)
    
        tool_map  = self.get_tools_map()
        tool_name = tool_call.function.name
                    
        if tool_name not in tool_map.keys():
            logger.error(f"Tool {tool_name} does not exist")
            return Message(
                role="tool",
                tool_call_id=tool_call.id,
                tool_name=tool_name,
                content=f"Tool {tool_name} does not exist",
            )

        try:
            function_args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse tool arguments: {e}")
            return Message(
                role="tool",
                tool_call_id=tool_call.id,
                tool_name=tool_name,
                content=f"Failed to parse tool arguments: {e}",
            )

        try:
            logger.debug(f"Tool call: {tool_name}({function_args})")
            observation = tool_map[tool_name].run(**function_args)
            if observation:
                return Message(
                    role="tool",
                    tool_call_id=tool_call.id,
                    tool_name=tool_name,
                    content=f"Observation: {observation}",
                )
        except Exception as e:
            logger.error(f"Error executing tool '{tool_name}': {e}")

        return Message(
            role="tool",
            tool_call_id=tool_call.id,
            tool_name=tool_name,
            content=f"Error executing tool '{tool_name}'",
        )


    def handle_tool_calls(self, tool_calls: List[ChatCompletionMessageToolCall]) -> List:
        
        tool_map = self.get_tools_map()
        if not tool_calls or not tool_map:
            return []
        
        messages = []
        for tool_call in tool_calls:
            observation = self.handle_tool_call(tool_call)
            if observation:
                messages.append(observation.to_dict())
            
        return messages

    async def ahandle_tool_calls(self, tool_calls: List[ChatCompletionMessageToolCall]) -> List:
        
        tool_map = self.get_tools_map()
        if not tool_calls or not tool_map:
            return []

        tasks = []
        for tool_call in tool_calls:
            task = asyncio.create_task(asyncio.to_thread(self.handle_tool_call, tool_call))
            tasks.append(task)

        messages = []

        results = await asyncio.gather(*tasks)
        for observation in results:
            if observation:
                messages.append(observation)

        return messages