import json
from typing import Any, Optional
from pydantic import Field, model_validator

from gwenflow.tools import BaseTool


class DuckDuckGoBaseTool(BaseTool):

    region: Optional[str] = "wt-wt"
    source: str = "text"
    time: Optional[str] = "y"
    max_results: int = 5
    safesearch: str = "moderate"
    backend: str = "api"

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Any) -> Any:
        """Validate that the python package exists in environment."""
        try:
            from duckduckgo_search import DDGS  # noqa: F401
        except ImportError:
            raise ImportError("duckduckgo-search is not installed. Please install it with `pip install duckduckgo-search`.")
        return values


class DuckDuckGoSearchTool(DuckDuckGoBaseTool):

    name: str = "DuckDuckGoSearchTool"
    description: str = "Search for a query in DuckDuckGo and returns the content."

    def _run(self, query: str = Field(description="The search query.")):
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = ddgs.text(
                query,
                region=self.region,  # type: ignore[arg-type]
                safesearch=self.safesearch,
                timelimit=self.time,
                max_results=self.max_results,
                backend=self.backend,
            )
            return json.dumps(results)


class DuckDuckGoNewsTool(DuckDuckGoBaseTool):

    name: str = "DuckDuckGoNewsTool"
    description: str = "Search for a query in DuckDuckGo News and returns the content."

    def _run(self, query: str = Field(description="The search query.")) -> str:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = ddgs.news(
                query,
                region=self.region,  # type: ignore[arg-type]
                safesearch=self.safesearch,
                timelimit=self.time,
                max_results=self.max_results,
            )
            return json.dumps(results)
