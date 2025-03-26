from typing import Any, Dict

from gwenflow.llms.openai import ChatOpenAI

class ChatOllama(ChatOpenAI):
 
    base_url: str

    def _get_client_params(self) -> Dict[str, Any]:

        client_params = {
            "base_url": self.base_url,
            "api_key": self.api_key,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }

        client_params = {k: v for k, v in client_params.items() if v is not None}

        return client_params