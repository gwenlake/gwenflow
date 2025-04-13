import asyncio
import uuid

from abc import ABC, abstractmethod
from typing import Any
from pydantic import BaseModel, model_validator

from gwenflow.tools.utils import function_to_json
from gwenflow.types.output import ResponseOutputItem
from gwenflow.logger import logger


class BaseTool(BaseModel, ABC):

    name: str
    """The unique name of the tool that clearly communicates its purpose."""

    description: str
    """Used to tell the model how to use the tool."""

    params_json_schema: dict[str, Any] = None
    """The JSON schema for the tool's parameters."""

    tool_type: str = "base"
    """Tool type: base, function, langchain."""

    max_results: int = 50
    """A max result for the tools."""

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

    def _response_to_list(self, data) -> list:
        if isinstance(data, str):
            return [ {"content": data} ]
        if isinstance(data, dict):
            return [ data ]
        if isinstance(data, list):
            return data
        logger.warning("Cannot read tool output.")
        return []
        
    @abstractmethod
    def _run(self, **kwargs: Any) -> Any:
        """Actual implementation of the tool."""

    def run(self, **kwargs: Any) -> ResponseOutputItem:
        response = self._run(**kwargs)
        return ResponseOutputItem(id=str(uuid.uuid4()), name=self.name, data=self._response_to_list(response))

    async def arun(self, **kwargs: Any) -> ResponseOutputItem:
        response = asyncio.run(self._run(**kwargs))
        return ResponseOutputItem(id=str(uuid.uuid4()), name=self.name, data=self._response_to_list(response))
