from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from typing import Any, Dict, List, Optional, cast
import os
import requests

from gwenflow.types import Document
from gwenflow.reranker.base import Reranker


class GwenlakeReranker(Reranker):
    """Gwenlake reranker."""

    api_base: str = "https://api.gwenlake.com/v1/rerank"
    api_key: Optional[SecretStr] = None
    
    model_config = ConfigDict(
        extra="forbid",
    )


    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Dict:
        values["api_key"] = os.getenv("GWENLAKE_API_KEY")
        if "model" not in values:
            values["model"] = "BAAI/bge-reranker-v2-m3"
        return values

    def _rerank(self, query: str, input: List[str]) -> List[List[float]]:

        api_key = cast(SecretStr, self.api_key).get_secret_value()

        payload = {"query": query, "input": input, "model": self.model}

        # HTTP headers for authorization
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        # send request
        try:
            response = requests.post(self.api_base, headers=headers, json=payload)
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Error raised by inference endpoint: {e}")
        
        if response.status_code != 200:
            raise ValueError(
                f"Error raised by inference API: rate limit exceeded.\nResponse: "
                f"{response.text}"
            )

        parsed_response = response.json()
        if "data" not in parsed_response:
            raise ValueError("Error raised by inference API.")

        reranking = []
        for e in parsed_response["data"]:
            reranking.append(e)
        
        return reranking

    def rerank(self, query: str, documents: List[Document]) -> List[Document]:

        if not documents:
            return []
        
        batch_size = 100
        reranked_documents = []
        try:
            for i in range(0, len(documents), batch_size):
                i_end = min(len(documents), i+batch_size)
                batch = documents[i:i_end]
                batch_processed = []
                for document in batch:
                    batch_processed.append(document.content)
                reranked_documents += self._rerank(query=query, input=batch_processed)
        except Exception as e:
            print(repr(e))
            return None

        if len(reranked_documents) > 0:

            compressed_documents = documents.copy()

            for i, _ in enumerate(compressed_documents):
                compressed_documents[i].score = reranked_documents[i]["relevance_score"]

            # Order by relevance score
            compressed_documents.sort(
                key=lambda x: x.score if x.score is not None else float("-inf"),
                reverse=True,
            )

            if self.top_k is not None:
                compressed_documents = compressed_documents[:self.top_k]

            if self.threshold is not None:
                compressed_documents = [d for d in compressed_documents if d.score > self.threshold]

            return compressed_documents

        return []
