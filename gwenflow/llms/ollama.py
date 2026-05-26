from dataclasses import dataclass
from typing import Any, Dict

from gwenflow.llms.openai import ChatOpenAI


@dataclass(kw_only=True)
class ChatOllama(ChatOpenAI):
    base_url: str

    # Local models (Gemma 3, Qwen3, DeepSeek-R1 distills) typically emit
    # reasoning inline as <think>...</think> rather than in a dedicated field.
    extract_think_tags: bool = True

    def _get_client_params(self) -> Dict[str, Any]:
        client_params = {
            "base_url": self.base_url,
            "api_key": self.api_key,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }

        client_params = {k: v for k, v in client_params.items() if v is not None}

        return client_params
