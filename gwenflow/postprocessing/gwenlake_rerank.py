from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from typing import Any, Dict, List, Optional, cast
import os
import requests


class GwenlakeRerank(BaseModel):
    """Gwenlake reranking models."""

    model: str
    top_k: int
    threshold: float

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

    def rerank_documents(self, query: str, texts: List[str]) -> List[List[float]]:
        """Call out to Gwenlake's embedding endpoint.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        batch_size = 100
        reranked_documents = []
        try:
            for i in range(0, len(texts), batch_size):
                i_end = min(len(texts), i+batch_size)
                batch = texts[i:i_end]
                batch_processed = []
                for text in batch:
                    batch_processed.append(text)
                reranked_documents += self._rerank(query=query, input=batch_processed)
        except Exception as e:
            print(repr(e))
            return None

        # filter if threshold
        if self.threshold is not None:
            reranked_documents = [doc for doc in reranked_documents if doc["relevance_score"] > self.threshold]

        if len(reranked_documents) > 0:
            reranked_documents = sorted(reranked_documents, key=lambda d: d['relevance_score'], reverse=True)
            reranked_documents = [x["text"] for x in reranked_documents]
            return reranked_documents[:self.top_k]

        return []
