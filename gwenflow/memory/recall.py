"""Automatic recall: pick the topic files a request is about, without an LLM call.

The agent could read them itself with ``read_memory``, but that costs a model
round trip per file. Requests usually name what they are about, so matching
the request text against the index (file names and descriptions) recalls the
right files at no cost; ``read_memory`` stays as the fallback for whatever the
text does not name.

Each index entry is scored against the request:

- its file name found as a whole phrase: +3
- an identifier of its description (a token containing a digit, years excluded): +3
- a distinctive word of its file name: +2
- a distinctive word of the head of its description (before ``(``, ``,`` or ``;``): +2
- any other distinctive word of its description: +1

"Distinctive" means at least 3 characters (4 for description words), not a
number, and not a stop word. Entries scoring at least the threshold (2) are
recalled, best first. Applications add their domain's generic words (legal
forms, the name of their own product, ...) through ``stopwords``.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable

from gwenflow.memory.store import MemoryFile

RECALL_THRESHOLD = 2

# Function words only (English and French); anything domain-specific belongs to the application.
DEFAULT_STOPWORDS = frozenset(
    {
        "the", "and", "for", "with", "from", "this", "that", "these", "those", "into", "about", "over",
        "les", "des", "une", "pour", "avec", "dans", "sur", "par", "est", "sont", "qui", "que", "aux",
        "chez", "mon", "mes", "ses", "leur", "leurs", "notre", "nos", "votre", "vos",
        "file", "files", "notes", "memory",
    }
)  # fmt: skip

_YEAR_RE = re.compile(r"^(19|20)\d{2}$")


def normalize_text(text: str) -> str:
    """Lowercase, strip accents, keep letters and digits separated by single spaces."""
    decomposed = unicodedata.normalize("NFKD", text or "")
    ascii_text = "".join(c for c in decomposed if not unicodedata.combining(c)).lower()
    return re.sub(r"[^a-z0-9]+", " ", ascii_text).strip()


def _distinctive(tokens: Iterable[str], stopwords: frozenset[str], min_len: int = 3) -> list[str]:
    return [t for t in tokens if len(t) >= min_len and t not in stopwords and not t.isdigit()]


def _identifiers(description: str) -> list[str]:
    """Id-like tokens of a description: contain a digit, 2+ characters, not a year."""
    ids = []
    for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]*", description or ""):
        if any(c.isdigit() for c in token) and len(token) >= 2 and not _YEAR_RE.match(token):
            normalized = normalize_text(token)
            if normalized:
                ids.append(normalized)
    return ids


@dataclass
class RecallHit:
    path: str
    score: int
    reasons: list[str]


def score_entry(
    entry: MemoryFile,
    normalized_text: str,
    tokens: set[str],
    stopwords: frozenset[str] = DEFAULT_STOPWORDS,
) -> RecallHit:
    padded = f" {normalized_text} "
    name = normalize_text(entry.path.rsplit("/", 1)[-1][: -len(".md")])
    description = entry.resolved_description()
    score = 0
    reasons: list[str] = []

    name_tokens = set(_distinctive(name.split(), stopwords))
    # The whole name counts as a phrase only when it holds at least one distinctive word.
    if name_tokens and f" {name} " in padded:
        score += 3
        reasons.append(f"name:{name}")
    for ident in _identifiers(description):
        if f" {ident} " in padded:
            score += 3
            reasons.append(f"id:{ident}")
    for token in name_tokens & tokens:
        score += 2
        reasons.append(f"name-word:{token}")
    head = re.split(r"[(,;]", description, maxsplit=1)[0] if description else ""
    head_tokens = set(_distinctive(normalize_text(head).split(), stopwords, min_len=4)) - name_tokens
    for token in head_tokens & tokens:
        score += 2
        reasons.append(f"desc-head:{token}")
    other = set(_distinctive(normalize_text(description).split(), stopwords, min_len=4)) - name_tokens - head_tokens
    for token in other & tokens:
        score += 1
        reasons.append(f"desc:{token}")
    return RecallHit(path=entry.path, score=score, reasons=reasons)


def select_relevant(
    entries: Iterable[MemoryFile],
    text: str,
    *,
    limit: int = 3,
    threshold: int = RECALL_THRESHOLD,
    stopwords: Iterable[str] = (),
) -> list[RecallHit]:
    """The entries ``text`` is about, best first, at most ``limit``.

    ``stopwords`` are added to :data:`DEFAULT_STOPWORDS` (compared after
    :func:`normalize_text`).
    """
    if not text or limit <= 0:
        return []
    words = DEFAULT_STOPWORDS | frozenset(normalize_text(w) for w in stopwords)
    normalized = normalize_text(text)
    tokens = set(normalized.split())
    hits = [score_entry(e, normalized, tokens, words) for e in entries]
    hits = [h for h in hits if h.score >= threshold]
    hits.sort(key=lambda h: (-h.score, h.path))
    return hits[:limit]
