import os
from typing import Any, Optional
from pydantic import Field, model_validator

from gwenflow.tools import BaseTool
from gwenflow.utils import logger


class TavilyBaseTool(BaseTool):

    client: Any
    api_key: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Any) -> Any:
        """Validate that the python package exists in environment."""
        try:
            from tavily import TavilyClient
            api_key = values["api_key"] or os.getenv("TAVILY_API_KEY")
            if not api_key:
                logger.error("TAVILY_API_KEY not provided")
            values["client"] = TavilyClient(api_key=api_key)
        except ImportError:
            raise ImportError("`tavily-python` not installed. Please install using `pip install tavily-python`")
        return values


class TavilyWebSearchTool(TavilyBaseTool):

    name: str = "TavilyWebSearchTool"
    description: str = "Use this function to search the web for a given query."

    def _run(self, query: str = Field(description="Query to search for.")) -> str:
        search_depth = "advanced"
        max_tokens = 6000
        return self.client.get_search_context(query=query, search_depth=search_depth, max_tokens=max_tokens)
