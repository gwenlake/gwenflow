from dataclasses import dataclass
from typing import Optional

from pydantic import Field

from gwenflow.tools.tool import BaseTool


@dataclass(kw_only=True)
class DuckDuckGoTool(BaseTool):
    region: Optional[str] = "wt-wt"
    source: str = "text"
    time: Optional[str] = "y"
    max_results: int = 5
    safesearch: str = "moderate"
    backend: str = "api"

    def __post_init__(self) -> None:
        try:
            from duckduckgo_search import DDGS  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "duckduckgo-search is not installed. Please install it with `pip install duckduckgo-search`."
            ) from e
        super().__post_init__()


@dataclass(kw_only=True)
class DuckDuckGoSearchTool(DuckDuckGoTool):
    name: str = "DuckDuckGoSearchTool"
    description: str = "Search for a query in DuckDuckGo and returns the content."

    def _run(self, query: str = Field(description="The search query.")):
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            return ddgs.text(
                query,
                region=self.region,
                safesearch=self.safesearch,
                timelimit=self.time,
                max_results=self.max_results,
                backend=self.backend,
            )


@dataclass(kw_only=True)
class DuckDuckGoNewsTool(DuckDuckGoTool):
    name: str = "DuckDuckGoNewsTool"
    description: str = "Search for a query in DuckDuckGo News and returns the content."

    def _run(self, query: str = Field(description="The search query.")):
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            return ddgs.news(
                query,
                region=self.region,
                safesearch=self.safesearch,
                timelimit=self.time,
                max_results=self.max_results,
            )
