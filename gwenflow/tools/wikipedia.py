from pydantic import Field

from gwenflow.tools import BaseTool
from gwenflow.utilities.wikipedia import WikipediaWrapper


class Wikipedia(BaseTool):

    name: str = "wikipedia"
    description: str = (
        "A wrapper around Wikipedia. "
        "Useful for when you need to answer general questions about "
        "people, places, companies, facts, historical events, or other subjects. "
        "Input should be a search query."
    )

    def _run(self, query: str = Field(description="query to look up on wikipedia")):
        return WikipediaWrapper().run(query)
