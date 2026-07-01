import dataclasses

from gwenflow.types import Document


def reciprocal_rank_fusion(ranked_lists: list[list[Document]], k: int = 60) -> list[Document]:
    """Fuse several ranked document lists into one via Reciprocal Rank Fusion.

    Each input list must already be sorted best match first. Documents are
    matched across lists by `id` rather than by position, so the lists may be
    partial, differently ordered, or come from unrelated sources.
    """
    scores: dict[str, float] = {}
    documents: dict[str, Document] = {}

    for ranked in ranked_lists:
        for rank, document in enumerate(ranked, start=1):
            scores[document.id] = scores.get(document.id, 0.0) + 1 / (k + rank)
            documents.setdefault(document.id, document)

    fused = [dataclasses.replace(documents[document_id], score=score) for document_id, score in scores.items()]
    fused.sort(key=lambda document: document.score, reverse=True)
    return fused
