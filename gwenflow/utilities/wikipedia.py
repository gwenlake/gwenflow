from typing import Any, Dict, Iterator, List, Optional
from pydantic import BaseModel, model_validator

from gwenflow.utils import logger


WIKIPEDIA_MAX_QUERY_LENGTH = 300


class WikipediaWrapper(BaseModel):

    wiki_client: Any
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
            values["wiki_client"] = wikipedia
        except ImportError:
            raise ImportError(
                "Could not import wikipedia python package. "
                "Please install it with `pip install wikipedia`."
            )
        return values

    def run(self, query: str) -> str:
        try:
            import wikipedia  # noqa: F401
        except ImportError:
            raise ImportError(
                "The `wikipedia` package is not installed. " "Please install it via `pip install wikipedia`."
            )

        page_titles = self.wiki_client.search(
            query[:WIKIPEDIA_MAX_QUERY_LENGTH], results=self.top_k_results
        )

        summaries = []
        for page_title in page_titles[: self.top_k_results]:
            try:
                wiki_page = self.wiki_client.page(title=page_title, auto_suggest=False)
                summary = f"Page: {page_title}\nSummary: {wiki_page.summary}"
                summaries.append(summary)
            except Exception as e:
                pass

        if not summaries:
            return "No good Wikipedia Search Result was found"

        return "\n\n".join(summaries)[: self.doc_content_chars_max]
