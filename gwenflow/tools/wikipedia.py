import json
from pydantic import Field

from gwenflow.documents import Document
from gwenflow.tools import BaseTool
from gwenflow.utils import logger


class Wikipedia(BaseTool):

    name: str = "wikipedia"
    description: str = (
        "A wrapper around Wikipedia. "
        "Useful for when you need to answer general questions about "
        "people, places, companies, facts, historical events, or other subjects. "
        "Input should be a search query."
    )

    def _run(self, query: str = Field(description="query to look up on wikipedia")):
        try:
            import wikipedia  # noqa: F401
        except ImportError:
            raise ImportError(
                "The `wikipedia` package is not installed. " "Please install it via `pip install wikipedia`."
            )

        logger.info(f"Searching wikipedia for: {query}")
        return json.dumps(Document(name=query, content=wikipedia.summary(query)).to_dict())
