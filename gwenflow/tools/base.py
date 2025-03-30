from abc import ABC, abstractmethod
from typing import Any
from pydantic import BaseModel, model_validator

from gwenflow.tools.utils import function_to_json


class BaseTool(BaseModel, ABC):

    name: str
    """The unique name of the tool that clearly communicates its purpose."""

    description: str
    """Used to tell the model how to use the tool."""

    params_json_schema: dict[str, Any] = None
    """The JSON schema for the tool's parameters."""

    tool_type: str = "base"
    """Tool type: base, function, langchain."""

    @model_validator(mode="after")
    def model_valid(self) -> Any:
        if not self.params_json_schema:
            _schema = function_to_json(self._run, name=self.name, description=self.description)
            self.params_json_schema = _schema["function"]["parameters"]
        return self
    
    def to_openai(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description or "",
                "parameters": self.params_json_schema,
            },
        }

    @abstractmethod
    def _run(self, **kwargs: Any) -> Any:
        """Actual implementation of the tool."""

    def run(self, **kwargs: Any) -> Any:
        return self._run(**kwargs)

