import os
import json
# import dirtyjson
from typing import Any, Optional, Dict
from pydantic import Field, model_validator

from gwenflow.logger import logger
from gwenflow.tools import BaseTool


class TavilyBaseTool(BaseTool):

    client: Optional[Any] = None
    api_key: Optional[str] = None
    max_tokens: int = 6000

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
    description: str = "Use this function to search Google for fully-formed URL to enhance your knowledge."

    def _run(self, query: str = Field(description="Query to search for.")):

        search_depth = "advanced"
        max_tokens = 6000

        response = self.client.search(query=query, search_depth=search_depth, max_tokens=max_tokens)

        clean_results = []
        current_token_count = 0
        for result in response.get("results", []):
            _result = {
                "title": result["title"],
                "url": result["url"],
                "content": result["content"],
                "score": result["score"],
            }
            current_token_count += len(json.dumps(_result))
            if current_token_count > self.max_tokens:
                break
            clean_results.append(_result)
        
        return clean_results
