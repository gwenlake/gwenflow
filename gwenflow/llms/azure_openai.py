import os
from typing import Optional, Union

import openai

from gwenflow.llms.openai import ChatOpenAI


class ChatAzureOpenAI(ChatOpenAI):
 
    def __init__(
        self,
        *,
        azure_endpoint: Optional[str] = None,
        azure_deployment: Optional[str] = None,
        api_version: Optional[str] = None,
        api_key: Optional[str] = None,
        model: str,
        temperature: Optional[float] = 0.0,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        ):
        _api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        _azure_deployment = azure_deployment or os.environ.get("AZURE_OPENAI_DEPLOYMENT")
        _azure_endpoint = azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        _api_version = api_version or os.environ.get("AZURE_OPENAI_API_VERSION")

        self.client = openai.AzureOpenAI(
            azure_endpoint=_azure_endpoint,
            azure_deployment=_azure_deployment,
            api_version=_api_version,
            api_key=_api_key,
        )

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p