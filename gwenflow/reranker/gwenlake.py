from dataclasses import dataclass
from functools import cached_property
from typing import List, Optional

import requests

from gwenflow.api import Api, api
from gwenflow.logger import logger
from gwenflow.reranker.base import Reranker
from gwenflow.telemetry import tracer
from gwenflow.types import Document


@dataclass(kw_only=True)
class GwenlakeReranker(Reranker):
    """Gwenlake reranker."""

    model: str = "BAAI/bge-reranker-v2-m3"
    base_url: Optional[str] = None

    @cached_property
    def _api(self) -> Api:
        return Api(base_url=self.base_url) if self.base_url else api

    def _rerank(self, query: str, input: List[str]) -> List[List[float]]:
        try:
            payload = {"query": query, "input": input, "model": self.model}
            response = self._api.client.post("/v1/rerank", json=payload)
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Error raised by inference endpoint: {e}") from e

        if response.status_code != 200:
            raise ValueError(f"Error raised by inference API: rate limit exceeded.\nResponse: {response.text}")

        parsed_response = response.json()
        if "data" not in parsed_response:
            raise ValueError("Error raised by inference API.")

        reranking = []
        for e in parsed_response["data"]:
            reranking.append(e)

        return reranking

    @tracer.reranker(name="Rerank")
    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        if not documents:
            return []

        batch_size = 100
        reranked_documents = []
        try:
            for i in range(0, len(documents), batch_size):
                i_end = min(len(documents), i + batch_size)
                batch = documents[i:i_end]
                batch_processed = []
                for document in batch:
                    batch_processed.append(document.content)
                reranked_documents += self._rerank(query=query, input=batch_processed)
        except Exception as e:
            logger.error(e)
            return None

        if len(reranked_documents) > 0:
            compressed_documents = documents.copy()

            for i, _ in enumerate(compressed_documents):
                compressed_documents[i].score = reranked_documents[i]["relevance_score"]

            compressed_documents.sort(
                key=lambda x: x.score if x.score is not None else float("-inf"),
                reverse=True,
            )

            if self.top_k is not None:
                compressed_documents = compressed_documents[: self.top_k]

            if self.threshold is not None:
                compressed_documents = [d for d in compressed_documents if d.score > self.threshold]

            return compressed_documents

        return []
