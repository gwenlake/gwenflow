from typing import Any, Dict, Iterator, List, Optional
from pydantic import BaseModel, model_validator

from gwenflow.utils import logger



class DuckDuckGoSearchWrapper(BaseModel):

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

    def _search_text(self, query: str, max_results: Optional[int] = None) -> List[Dict[str, str]]:
        """Run query through DuckDuckGo text search and return results."""
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            return ddgs.text(
                query,
                region=self.region,  # type: ignore[arg-type]
                safesearch=self.safesearch,
                timelimit=self.time,
                max_results=max_results or self.max_results,
                backend=self.backend,
            )
    
    def _search_news(self, query: str, max_results: Optional[int] = None) -> List[Dict[str, str]]:
        """Run query through DuckDuckGo news search and return results."""
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            return ddgs.news(
                query,
                region=self.region,  # type: ignore[arg-type]
                safesearch=self.safesearch,
                timelimit=self.time,
                max_results=max_results or self.max_results,
            )

    def search(self, query: str, max_results: Optional[int] = None, source: Optional[str] = None) -> List[Dict[str, str]]:
        """Run query through DuckDuckGo and return metadata.

        Args:
            query: The query to search for.
            max_results: The number of results to return.
            source: The source to look from.

        Returns:
            A list of dictionaries with the following keys:
                content - The description of the result.
                title - The title of the result.
                link - The link to the result.
        """

        results = []

        source = source or self.source

        if source == "text":
            results = self._search_text(query, max_results=max_results)

        elif source == "news":
            results = self._search_news(query, max_results=max_results)

        return results
