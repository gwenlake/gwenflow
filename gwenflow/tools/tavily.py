import os
from dataclasses import dataclass
from typing import Any, Optional

from pydantic import Field

from gwenflow.logger import logger
from gwenflow.tools.tool import Tool


@dataclass(kw_only=True)
class TavilyTool(Tool):
    client: Optional[Any] = None
    api_key: Optional[str] = None
    max_results: int = 5
    search_depth: str = "advanced"

    def __post_init__(self) -> None:
        try:
            from tavily import TavilyClient

            if self.client is None:
                if self.api_key is None:
                    self.api_key = os.getenv("TAVILY_API_KEY")
                if self.api_key is None:
                    logger.error("TAVILY_API_KEY not provided")
                self.client = TavilyClient(api_key=self.api_key)
        except ImportError as e:
            raise ImportError("`tavily-python` is not installed. Please install it with `uv add tavily-python`") from e
        super().__post_init__()


@dataclass(kw_only=True)
class TavilyWebSearchTool(TavilyTool):
    name: str = "TavilyWebSearchTool"
    description: str = "Use this function to search Google for fully-formed URL to enhance your knowledge."

    def _run(self, query: str = Field(description="Query to search for.")):
        response = self.client.search(query=query, search_depth=self.search_depth, max_results=self.max_results)
        return [
            {
                "title": r["title"],
                "url": r["url"],
                "content": r["content"],
                "score": r["score"],
            }
            for r in response.get("results", [])
        ]
