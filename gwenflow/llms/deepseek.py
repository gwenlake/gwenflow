import os
from typing import Dict, Any

from gwenflow.llms.openai import ChatOpenAI


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
     