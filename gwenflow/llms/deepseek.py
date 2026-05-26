import os
from dataclasses import dataclass
from typing import Any, Dict

from gwenflow.llms.openai import ChatOpenAI, response_to_openai_dict
from gwenflow.types import ModelResponse


def response_to_deepseek_dict(response: ModelResponse) -> Dict[str, Any]:
    """Serialize a ModelResponse to a DeepSeek ChatCompletion-shaped dict.

    DeepSeek's API is OpenAI-compatible but exposes reasoning under
    `reasoning_content` (not `reasoning`).
    """
    completion = response_to_openai_dict(response)
    message = completion["choices"][0]["message"]
    if "reasoning" in message:
        message["reasoning_content"] = message.pop("reasoning")
    return completion


@dataclass(kw_only=True)
class ChatDeepSeek(ChatOpenAI):
    model: str = "deepseek-chat"
    base_url: str = "https://api.deepseek.com"

    def _get_client_params(self) -> Dict[str, Any]:
        api_key = self.api_key
        if api_key is None:
            api_key = os.environ.get("DEEPSEEK_API_KEY")
        if api_key is None:
            raise ValueError(
                "The api_key client option must be set either by passing api_key to the client or by setting the DEEPSEEK_API_KEY environment variable"
            )

        client_params = {
            "api_key": api_key,
            "base_url": self.base_url,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }

        client_params = {k: v for k, v in client_params.items() if v is not None}

        return client_params
