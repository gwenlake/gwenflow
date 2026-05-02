import os
from unittest.mock import MagicMock

import pytest

from gwenflow.embeddings.gwenlake import EMBEDDING_DIMS, GwenlakeEmbeddings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_api(embeddings):
    """Return a mock Api whose client.post returns the given embeddings."""
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.json.return_value = {"data": [{"embedding": e} for e in embeddings]}
    fake_client = MagicMock()
    fake_client.post.return_value = fake_response
    fake_api = MagicMock()
    fake_api.client = fake_client
    return fake_api


def _capturing_fake_api(embeddings):
    """Fake API that also records what was posted."""
    captured = []
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.json.return_value = {"data": [{"embedding": e} for e in embeddings]}

    fake_client = MagicMock()

    def capture(path, json):
        captured.append(json["input"])
        return fake_response

    fake_client.post.side_effect = capture
    fake_api = MagicMock()
    fake_api.client = fake_client
    return fake_api, captured


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def test_default_model():
    emb = GwenlakeEmbeddings()
    assert emb.model == "intfloat/e5-base-v2"


def test_default_dimensions_derived_from_model():
    emb = GwenlakeEmbeddings()
    assert emb.dimensions == EMBEDDING_DIMS["intfloat/e5-base-v2"]


def test_custom_model_sets_correct_dimensions():
    emb = GwenlakeEmbeddings(model="intfloat/e5-large-v2")
    assert emb.dimensions == 1024


def test_multilingual_model_dimensions():
    emb = GwenlakeEmbeddings(model="intfloat/multilingual-e5-base")
    assert emb.dimensions == 768


def test_base_url_defaults_to_none():
    emb = GwenlakeEmbeddings()
    assert emb.base_url is None


def test_custom_base_url_passed_to_api():
    from gwenflow.api import Api

    emb = GwenlakeEmbeddings(base_url="https://custom.example.com")
    assert isinstance(emb._api, Api)
    assert emb._api.base_url == "https://custom.example.com"


# ---------------------------------------------------------------------------
# embed_documents
# ---------------------------------------------------------------------------


def test_embed_documents_returns_embeddings():
    emb = GwenlakeEmbeddings()
    emb.__dict__["_api"] = _make_fake_api([[0.1, 0.2], [0.3, 0.4]])

    result = emb.embed_documents(["hello", "world"])

    assert len(result) == 2
    assert result[0] == [0.1, 0.2]
    assert result[1] == [0.3, 0.4]


def test_embed_documents_empty_returns_empty():
    emb = GwenlakeEmbeddings()
    assert emb.embed_documents([]) == []


def test_embed_documents_adds_passage_prefix():
    emb = GwenlakeEmbeddings()
    fake_api, captured = _capturing_fake_api([[0.1]])
    emb.__dict__["_api"] = fake_api

    emb.embed_documents(["some text"])

    assert captured[0][0].startswith("passage: ")


def test_embed_documents_no_double_prefix():
    emb = GwenlakeEmbeddings()
    fake_api, captured = _capturing_fake_api([[0.1]])
    emb.__dict__["_api"] = fake_api

    emb.embed_documents(["passage: already prefixed"])

    assert captured[0][0] == "passage: already prefixed"


def test_embed_documents_strips_extra_whitespace():
    emb = GwenlakeEmbeddings(model="intfloat/e5-large-v2")
    fake_api, captured = _capturing_fake_api([[0.1]])
    emb.__dict__["_api"] = fake_api

    emb.embed_documents(["  hello   world  "])

    text = captured[0][0]
    assert not text.startswith(" ")
    assert "  " not in text


# ---------------------------------------------------------------------------
# embed_query
# ---------------------------------------------------------------------------


def test_embed_query_returns_vector():
    emb = GwenlakeEmbeddings()
    emb.__dict__["_api"] = _make_fake_api([[0.5, 0.6, 0.7]])

    result = emb.embed_query("hello")

    assert result == [0.5, 0.6, 0.7]


def test_embed_query_adds_query_prefix():
    emb = GwenlakeEmbeddings()
    fake_api, captured = _capturing_fake_api([[0.1]])
    emb.__dict__["_api"] = fake_api

    emb.embed_query("some text")

    assert captured[0][0].startswith("query: ")


def test_embed_query_no_double_prefix():
    emb = GwenlakeEmbeddings()
    fake_api, captured = _capturing_fake_api([[0.1]])
    emb.__dict__["_api"] = fake_api

    emb.embed_query("query: already prefixed")

    assert captured[0][0] == "query: already prefixed"


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("GWENLAKE_API_KEY"), reason="GWENLAKE_API_KEY missing")
def test_embed_documents_real_api():
    emb = GwenlakeEmbeddings()
    result = emb.embed_documents(["hello world"])

    assert result is not None
    assert len(result) == 1
    assert len(result[0]) == EMBEDDING_DIMS["intfloat/e5-base-v2"]


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("GWENLAKE_API_KEY"), reason="GWENLAKE_API_KEY missing")
def test_embed_query_real_api():
    emb = GwenlakeEmbeddings()
    result = emb.embed_query("hello world")

    assert result is not None
    assert len(result) == EMBEDDING_DIMS["intfloat/e5-base-v2"]
