from dataclasses import dataclass
from typing import Any, Optional

from pydantic import Field

from gwenflow.tools.tool import BaseTool

WIKIPEDIA_MAX_QUERY_LENGTH = 300


@dataclass(kw_only=True)
class WikipediaBase(BaseTool):
    client: Optional[Any] = None
    lang: str = "en"
    top_k_results: int = 5
    doc_content_chars_max: int = 4000

    def __post_init__(self) -> None:
        try:
            import wikipedia

            wikipedia.set_lang(self.lang)
            self.client = wikipedia
        except ImportError as e:
            raise ImportError(
                "Could not import wikipedia python package. Please install it with `pip install wikipedia`."
            ) from e
        super().__post_init__()


@dataclass(kw_only=True)
class WikipediaTool(WikipediaBase):
    name: str = "WikipediaTool"
    description: str = (
        "A wrapper around Wikipedia. "
        "Useful for when you need to answer general questions about "
        "people, places, companies, facts, historical events, or other subjects. "
        "Input should be a search query."
    )

    def _run(self, query: str = Field(description="query to look up on wikipedia")):
        documents = []
        doc_content_chars = 0
        page_titles = self.client.search(query[:WIKIPEDIA_MAX_QUERY_LENGTH], results=self.top_k_results)
        for page_title in page_titles[: self.top_k_results]:
            try:
                page = self.client.page(title=page_title, auto_suggest=False)
                summary = {"title": page_title, "summary": page.summary}
                if (doc_content_chars + len(page_title) + len(page.summary) + 20) < self.doc_content_chars_max:
                    documents.append(summary)
                    doc_content_chars += len(page_title) + len(page.summary) + 20
            except Exception:
                pass

        if not documents:
            raise ValueError(f"No Wikipedia results found for query: {query!r}")
        return list(documents)
