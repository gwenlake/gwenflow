from typing import Any, Dict
import os
import logging

from gwenflow.llms.openai import ChatOpenAI


logger = logging.getLogger(__name__)


class ChatMistral(ChatOpenAI):
 
    model: str = "open-mistral-7b"
    base_url: str = "https://api.mistral.ai"

    def _get_client_params(self) -> Dict[str, Any]:

        api_key = self.api_key
        if api_key is None:
            api_key = os.environ.get("MISTRAL_API_KEY")
        if api_key is None:
            raise ValueError(
                "The api_key client option must be set either by passing api_key to the client or by setting the MISTRAL_API_KEY environment variable"
            )

        client_params = {
            "api_key": api_key,
            "base_url": self.base_url,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }

        client_params = {k: v for k, v in client_params.items() if v is not None}

        return client_params
     