import os
from unittest.mock import MagicMock

import pytest

from gwenflow.reranker.gwenlake import GwenlakeReranker
from gwenflow.types import Document


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _doc(content, score=None):
    return Document(content=content, score=score)


def _make_fake_api(scores):
    """Return a mock Api whose client.post returns the given relevance scores."""
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.json.return_value = {"data": [{"relevance_score": s} for s in scores]}
    fake_client = MagicMock()
    fake_client.post.return_value = fake_response
    fake_api = MagicMock()
    fake_api.client = fake_client
    return fake_api


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def test_default_model():
    r = GwenlakeReranker()
    assert r.model == "BAAI/bge-reranker-v2-m3"


def test_custom_model():
    r = GwenlakeReranker(model="my-reranker")
    assert r.model == "my-reranker"


def test_base_url_defaults_to_none():
    r = GwenlakeReranker()
    assert r.base_url is None


def test_custom_base_url_passed_to_api():
    from gwenflow.api import Api

    r = GwenlakeReranker(base_url="https://custom.example.com")
    assert isinstance(r._api, Api)
    assert r._api.base_url == "https://custom.example.com"


def test_top_k_defaults_to_none():
    r = GwenlakeReranker()
    assert r.top_k is None


def test_threshold_defaults_to_none():
    r = GwenlakeReranker()
    assert r.threshold is None


# ---------------------------------------------------------------------------
# rerank
# ---------------------------------------------------------------------------


def test_rerank_empty_documents():
    r = GwenlakeReranker()
    assert r.rerank("query", []) == []


def test_rerank_returns_sorted_by_score():
    r = GwenlakeReranker()
    r.__dict__["_api"] = _make_fake_api([0.3, 0.9, 0.1])

    docs = [_doc("low"), _doc("high"), _doc("lowest")]
    result = r.rerank("query", docs)

    assert result[0].content == "high"
    assert result[0].score == pytest.approx(0.9)
    assert result[1].score == pytest.approx(0.3)
    assert result[2].score == pytest.approx(0.1)


def test_rerank_preserves_all_documents_without_filters():
    r = GwenlakeReranker()
    r.__dict__["_api"] = _make_fake_api([0.5, 0.8, 0.2])

    docs = [_doc("a"), _doc("b"), _doc("c")]
    result = r.rerank("query", docs)

    assert len(result) == 3


def test_rerank_top_k_limits_results():
    r = GwenlakeReranker(top_k=2)
    r.__dict__["_api"] = _make_fake_api([0.3, 0.9, 0.1])

    docs = [_doc("a"), _doc("b"), _doc("c")]
    result = r.rerank("query", docs)

    assert len(result) == 2


def test_rerank_top_k_returns_highest_scores():
    r = GwenlakeReranker(top_k=2)
    r.__dict__["_api"] = _make_fake_api([0.3, 0.9, 0.1])

    docs = [_doc("low"), _doc("high"), _doc("lowest")]
    result = r.rerank("query", docs)

    scores = [d.score for d in result]
    assert 0.9 in scores
    assert 0.3 in scores


def test_rerank_threshold_filters_low_scores():
    r = GwenlakeReranker(threshold=0.5)
    r.__dict__["_api"] = _make_fake_api([0.3, 0.9, 0.1])

    docs = [_doc("a"), _doc("b"), _doc("c")]
    result = r.rerank("query", docs)

    assert len(result) == 1
    assert result[0].score == pytest.approx(0.9)


def test_rerank_threshold_all_filtered():
    r = GwenlakeReranker(threshold=0.99)
    r.__dict__["_api"] = _make_fake_api([0.3, 0.9, 0.1])

    docs = [_doc("a"), _doc("b"), _doc("c")]
    result = r.rerank("query", docs)

    assert result == []


def test_rerank_top_k_and_threshold_combined():
    r = GwenlakeReranker(top_k=3, threshold=0.5)
    r.__dict__["_api"] = _make_fake_api([0.3, 0.9, 0.6, 0.1])

    docs = [_doc("a"), _doc("b"), _doc("c"), _doc("d")]
    result = r.rerank("query", docs)

    # top_k=3 keeps top 3, then threshold removes those < 0.5 → 0.9 and 0.6
    assert len(result) == 2
    assert all(d.score >= 0.5 for d in result)


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("GWENLAKE_API_KEY"), reason="GWENLAKE_API_KEY missing")
def test_rerank_real_api():
    r = GwenlakeReranker()
    docs = [_doc("Paris is the capital of France."), _doc("The Eiffel Tower is in Paris."), _doc("Python is a language.")]
    result = r.rerank("Where is the Eiffel Tower?", docs)

    assert len(result) == 3
    assert result[0].score >= result[-1].score
