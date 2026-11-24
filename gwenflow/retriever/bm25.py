import dataclasses
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Callable, Iterable, Literal, Optional

from gwenflow.types import Document

TOKEN_PATTERN = re.compile(r"\w+")
CHAR_NGRAM_PREFIX = "#"

Variant = Literal["okapi", "plus", "l"]


@dataclass
class BM25Tokenizer:
    lowercase: bool = True
    strip_accents: bool = False
    stopwords: Optional[Iterable[str]] = None
    min_token_length: int = 1
    ngram_range: tuple[int, int] = (1, 1)
    char_ngrams: Optional[tuple[int, int]] = None
    stemmer: Optional[Callable[[str], str]] = None
    _stopwords: frozenset[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._check_range("ngram_range", self.ngram_range)
        if self.char_ngrams is not None:
            self._check_range("char_ngrams", self.char_ngrams)
        self._stopwords = frozenset(self._normalize(word) for word in self.stopwords or ())

    @staticmethod
    def _check_range(name: str, value: tuple[int, int]) -> None:
        low, high = value
        if low < 1 or high < low:
            raise ValueError(f"`{name}` must satisfy 1 <= min <= max, got {value}.")

    def _normalize(self, text: str) -> str:
        if self.lowercase:
            text = text.lower()
        if self.strip_accents:
            text = "".join(char for char in unicodedata.normalize("NFKD", text) if not unicodedata.combining(char))
        return text

    def _words(self, text: str) -> list[str]:
        words = [
            word
            for word in TOKEN_PATTERN.findall(self._normalize(text))
            if len(word) >= self.min_token_length and word not in self._stopwords
        ]
        if self.stemmer is not None:
            words = [self.stemmer(word) for word in words]
        return words

    def __call__(self, text: str) -> list[str]:
        words = self._words(text)

        low, high = self.ngram_range
        tokens = [" ".join(words[i : i + n]) for n in range(low, high + 1) for i in range(len(words) - n + 1)]

        if self.char_ngrams is not None:
            low, high = self.char_ngrams
            for word in words:
                padded = f"<{word}>"
                tokens.extend(
                    CHAR_NGRAM_PREFIX + padded[i : i + n]
                    for n in range(low, high + 1)
                    for i in range(len(padded) - n + 1)
                )

        return tokens


class BM25:
    """Lexical ranker over a fixed corpus of documents using BM25.

    The corpus is tokenized and indexed once at construction, so the same
    instance can rank many queries against it without rebuilding the index.
    """

    def __init__(
        self,
        documents: list[Document],
        variant: Variant = "okapi",
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: Optional[float] = None,
        delta: Optional[float] = None,
        tokenizer: Optional[Callable[[str], list[str]]] = None,
    ):
        try:
            from rank_bm25 import BM25L, BM25Okapi, BM25Plus
        except ImportError as exc:
            raise ImportError("`rank_bm25` is not installed.") from exc

        variants = {"okapi": BM25Okapi, "plus": BM25Plus, "l": BM25L}
        if variant not in variants:
            raise ValueError(f"Unknown BM25 variant `{variant}`, expected one of {sorted(variants)}.")
        if epsilon is not None and variant != "okapi":
            raise ValueError("`epsilon` only applies to the `okapi` variant.")
        if delta is not None and variant == "okapi":
            raise ValueError("`delta` only applies to the `plus` and `l` variants.")

        self.documents = documents
        self.tokenizer = tokenizer or BM25Tokenizer()

        params = {"k1": k1, "b": b}
        if epsilon is not None:
            params["epsilon"] = epsilon
        if delta is not None:
            params["delta"] = delta

        tokenized_corpus = [self.tokenizer(document.content) for document in documents]
        self._bm25 = variants[variant](tokenized_corpus, **params) if any(tokenized_corpus) else None

    def rank(self, query: str, top_n: Optional[int] = None, min_score: Optional[float] = None) -> list[Document]:
        """Rank the corpus against a query, best match first."""
        if self._bm25 is None:
            return []

        scores = self._bm25.get_scores(self.tokenizer(query))
        ranked = sorted(zip(self.documents, scores, strict=True), key=lambda pair: pair[1], reverse=True)
        if min_score is not None:
            ranked = [(document, score) for document, score in ranked if score > min_score]
        documents = [dataclasses.replace(document, score=float(score)) for document, score in ranked]

        if top_n is not None:
            documents = documents[:top_n]
        return documents
