import json
from pydantic import Field

from gwenflow.tools import BaseTool
from gwenflow.utilities.duckduckgo import DuckDuckGoSearchWrapper


class DuckDuckGoSearchTool(BaseTool):

    name: str = "duckduckgo-search"
    description: str = "This function search for a query in DuckDuckGo and returns the content."

    def _run(self, query: str = Field(description="The search query.")):
        reader = DuckDuckGoSearchWrapper()
        results = reader.search(query=query)
        return json.dumps(results)

class DuckDuckGoNewsTool(BaseTool):

    name: str = "duckduckgo-news"
    description: str = "This function search for a query in DuckDuckGo News and returns the content."

    def _run(self, query: str = Field(description="The search query.")):
        reader = DuckDuckGoSearchWrapper()
        results = reader.search(query=query, source="news")
        return json.dumps(results)
