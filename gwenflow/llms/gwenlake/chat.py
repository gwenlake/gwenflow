import os
from typing import Dict, Any

from gwenflow.llms.openai import ChatOpenAI


class ChatGwenlake(ChatOpenAI):
 
    model: str
    base_url: str = "https://api.gwenlake.com/v1"

    def _get_client_params(self) -> Dict[str, Any]:

        api_key = self.api_key
        if api_key is None:
            api_key = os.environ.get("GWENLAKE_API_KEY")
        if api_key is None:
            raise ValueError(
                "The api_key client option must be set either by passing api_key to the client or by setting the GWENLAKE_API_KEY environment variable"
            )

        organization = self.organization
        if organization is None:
            organization = os.environ.get('GWENLAKE_ORGANIZATION')

        client_params = {
            "api_key": api_key,
            "organization": organization,
            "base_url": self.base_url,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }

        client_params = {k: v for k, v in client_params.items() if v is not None}

        return client_params
        