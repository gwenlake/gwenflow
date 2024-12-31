import json
from typing import Any
from pydantic import Field, model_validator

from gwenflow.tools import BaseTool


WIKIPEDIA_MAX_QUERY_LENGTH = 300


class WikipediaBaseTool(BaseTool):

    client: Any
    lang: str = "en"
    top_k_results: int = 5
    doc_content_chars_max: int = 4000

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Any) -> Any:
        """Validate that the python package exists in environment."""
        try:
            import wikipedia
            lang = values.get("lang", "en")
            wikipedia.set_lang(lang)
            values["client"] = wikipedia
        except ImportError:
            raise ImportError(
                "Could not import wikipedia python package. "
                "Please install it with `pip install wikipedia`."
            )
        return values


class WikipediaTool(WikipediaBaseTool):

    name: str = "WikipediaTool"
    description: str = (
        "A wrapper around Wikipedia. "
        "Useful for when you need to answer general questions about "
        "people, places, companies, facts, historical events, or other subjects. "
        "Input should be a search query."
    )

    def _run(self, query: str = Field(description="query to look up on wikipedia")) -> str:

        documents = []

        doc_content_chars = 0
    
        page_titles = self.client.search(query[:WIKIPEDIA_MAX_QUERY_LENGTH], results=self.top_k_results)
        for page_title in page_titles[:self.top_k_results]:
            try:
                page = self.client.page(title=page_title, auto_suggest=False)
                summary = {"title": page_title, "summary": page.summary}
                if (doc_content_chars+len(page_title)+len(page.summary)+20) < self.doc_content_chars_max:
                    documents.append(summary)
                    doc_content_chars += len(page_title) + len(page.summary) + 20
            except Exception as e:
                pass

        if not documents:
            return "No good Wikipedia Search Result was found"

        return json.dumps([doc for doc in documents])
