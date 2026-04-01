from typing import Any, Callable

from gwenflow.tools.base import BaseTool
from gwenflow.tools.utils import function_to_json


class FunctionTool(BaseTool):
    func: Callable
    """The function that will be executed when the tool is called."""

    def _run(self, **kwargs: Any) -> Any:
        return self.func(**kwargs)

    @classmethod
    def from_function(cls, func: Callable) -> "FunctionTool":
        if func.__doc__ is None:
            raise ValueError("Function must have a docstring")
        if func.__annotations__ is None:
            raise ValueError("Function must have type annotations")

        openai_schema = function_to_json(func)

        return cls(
            name=func.__name__,
            description=func.__doc__,
            func=func,
            params_json_schema=openai_schema["function"]["parameters"],
            tool_type="function",
        )
