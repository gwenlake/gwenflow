from typing import Any, Callable

from langchain_core.tools import StructuredTool
from langchain_core.utils.function_calling import convert_to_openai_tool

from gwenflow.tools.utils import function_to_json
from gwenflow.tools.base import BaseTool


class FunctionTool(BaseTool):

    func: Callable
    """The function that will be executed when the tool is called."""

    def _run(self, **kwargs: Any) -> Any:
        if self.tool_type == "langchain":
            return self.func(kwargs)
        return self.func(**kwargs)
    
    @classmethod
    def from_function(cls, func: Callable) -> "FunctionTool":
        if func.__doc__ is None:
            raise ValueError("Function must have a docstring")
        if func.__annotations__ is None:
            raise ValueError("Function must have type annotations")
        openai_schema = function_to_json(func)
        return FunctionTool(
            name=func.__name__,
            description=func.__doc__,
            func=func,
            params_json_schema=openai_schema["function"]["parameters"],
            tool_type="function",
        )

    @classmethod
    def from_langchain(cls, tool: StructuredTool) -> "FunctionTool":
        if tool.run is None:
            raise ValueError("StructuredTool must have a callable 'func'")
        openai_schema=convert_to_openai_tool(tool)
        return FunctionTool(
            name=tool.name,
            description=tool.description,
            params_json_schema=openai_schema["function"]["parameters"],
            func=tool.run,
            tool_type="langchain",
        )
