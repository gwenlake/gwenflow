from dataclasses import dataclass
from typing import Any, Callable, Optional

from gwenflow.tools.tool import Tool, function_to_json_schema


@dataclass(kw_only=True)
class FunctionTool(Tool):
    func: Optional[Callable] = None

    def __post_init__(self) -> None:
        _schema = function_to_json_schema(self.func)
        self.name        = _schema["function"]["name"]
        self.description = _schema["function"]["description"]
        self.parameters  = _schema["function"]["parameters"]

    def _run(self, **kwargs: Any) -> Any:
        return self.func(**kwargs)

    @classmethod
    def from_function(cls, func: Callable) -> "FunctionTool":
        return cls(func=func)
