import os
from typing import Any, Optional
from pydantic import Field, model_validator

from gwenflow.tools import BaseTool
from gwenflow.utils import logger


class TavilyBaseTool(BaseTool):

    client: Optional[Any] = None
    api_key: Optional[str] = None

    @model_validator(mode="after")
    def validate_environment(self) -> 'TavilyBaseTool':
        """Validate that the python package exists in environment."""
        try:
            from tavily import TavilyClient
            if self.client is None:
                if self.api_key is None:
                    self.api_key = os.getenv("TAVILY_API_KEY")
                if self.api_key is None:
                    logger.error("TAVILY_API_KEY not provided")
                self.client = TavilyClient(api_key=self.api_key)
        except ImportError:
            raise ImportError("`tavily-python` not installed. Please install using `pip install tavily-python`")
        return self

class TavilyWebSearchTool(TavilyBaseTool):

    name: str = "TavilyWebSearchTool"
    description: str = "Use this function to search the web for a given query."

    def _run(self, query: str = Field(description="Query to search for.")) -> str:
        search_depth = "advanced"
        max_tokens = 6000
        return self.client.get_search_context(query=query, search_depth=search_depth, max_tokens=max_tokens)
