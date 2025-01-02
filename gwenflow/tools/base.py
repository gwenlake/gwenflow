from abc import ABC, abstractmethod
from typing import Any, Callable, Type
from pydantic import BaseModel, Field, model_validator, ConfigDict

from langchain_core.tools import StructuredTool
from langchain_core.utils.function_calling import convert_to_openai_tool

from gwenflow.tools.utils import function_to_json


class BaseTool(BaseModel, ABC):

    name: str
    """The unique name of the tool that clearly communicates its purpose."""

    description: str
    """Used to tell the model how to use the tool."""

    openai_schema: dict = None
    """OpenAI JSON schema"""

    tool_type: str = "function"
    """Tool type: function, langchain, llamaindex."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.openai_schema:
            self.openai_schema = function_to_json(self._run, name=self.name, description=self.description)

    @abstractmethod
    def _run(self, **kwargs: Any) -> Any:
        """Actual implementation of the tool."""

    def run(self, **kwargs: Any) -> Any:
        return self._run(**kwargs)
    
    

class Tool(BaseTool):

    func: Callable
    """The function that will be executed when the tool is called."""

    def _run(self, **kwargs: Any) -> Any:
        if self.tool_type == "langchain":
            return self.func(kwargs)
        return self.func(**kwargs)

    @classmethod
    def from_function(cls, func: Callable) -> "Tool":
        if cls == Tool:
            def _make_with_name(tool_name: str) -> Callable:
                def _make_tool(f: Callable) -> Tool:
                    if f.__doc__ is None:
                        raise ValueError("Function must have a docstring")
                    if f.__annotations__ is None:
                        raise ValueError("Function must have type annotations")
                    return Tool(
                        name=tool_name,
                        description=f.__doc__,
                        func=f,
                        openai_schema=function_to_json(f),
                        tool_type="function",
                    )
                return _make_tool

            if callable(func):
                return _make_with_name(func.__name__)(func)

            if isinstance(func, str):
                return _make_with_name(func)

        raise ValueError(f"Invalid arguments for {cls.__name__}")

    @classmethod
    def from_langchain(cls, tool: StructuredTool) -> "Tool":
        if cls == Tool:
            if tool.run is None:
                raise ValueError("StructuredTool must have a callable 'func'")
            return Tool(
                name=tool.name,
                description=tool.description,
                openai_schema=convert_to_openai_tool(tool),
                func=tool.run,
                tool_type="langchain",
            )
        raise NotImplementedError(f"from_langchain not implemented for {cls.__name__}")