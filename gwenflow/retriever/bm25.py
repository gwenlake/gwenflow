import dataclasses
from typing import Optional

from gwenflow.types import Document


class BM25:
    """Lexical ranker over a fixed corpus of documents using BM25.

    The corpus is tokenized and indexed once at construction, so the same
    instance can rank many queries against it without rebuilding the index.
    """

    def __init__(self, documents: list[Document]):
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise ImportError("`rank_bm25` is not installed.") from exc

        self.documents = documents
        tokenized_corpus = [document.content.lower().split() for document in documents]
        self._bm25 = BM25Okapi(tokenized_corpus) if documents else None

    def rank(self, query: str, top_n: Optional[int] = None) -> list[Document]:
        """Rank the corpus against a query, best match first."""
        if self._bm25 is None:
            return []

        scores = self._bm25.get_scores(query.lower().split())
        ranked = sorted(zip(self.documents, scores, strict=True), key=lambda pair: pair[1], reverse=True)
        documents = [dataclasses.replace(document, score=score) for document, score in ranked]

        if top_n is not None:
            documents = documents[:top_n]
        return documents
