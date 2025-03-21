import os
from typing import Optional, Dict, Any

from openai import AzureOpenAI, AsyncAzureOpenAI

from gwenflow.llms.openai import ChatOpenAI


class ChatAzureOpenAI(ChatOpenAI):
 
    api_version: Optional[str] = None
    azure_endpoint: Optional[str] = None
    azure_deployment: Optional[str] = None

    def _get_client_params(self) -> Dict[str, Any]:

        api_key = self.api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        azure_endpoint = self.azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        azure_deployment = self.azure_deployment or os.environ.get("AZURE_OPENAI_DEPLOYMENT")
        api_version = self.api_version or os.environ.get("AZURE_OPENAI_API_VERSION")

        if api_key is None:
            raise ValueError(
                "The api_key client option must be set either by passing api_key to the client or by setting the AZURE_OPENAI_API_KEY environment variable"
            )

        client_params = {
            "api_key": api_key,
            "api_version": api_version,
            "azure_endpoint": azure_endpoint,
            "azure_deployment": azure_deployment,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }
        
        client_params = {k: v for k, v in client_params.items() if v is not None}
        
        return client_params

    def get_client(self) -> AzureOpenAI:

        if self.client:
            return self.client
        
        client_params = self._get_client_params()

        self.client = AzureOpenAI(**client_params)
        return self.client

    def get_async_client(self) -> AsyncAzureOpenAI:

        if self.async_client:
            return self.async_client
        
        client_params = self._get_client_params()

        self.async_client = AsyncAzureOpenAI(**client_params)
        return self.async_client
    