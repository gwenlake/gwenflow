import os
from typing import Optional, Union

import openai

from gwenflow.llms.openai import ChatOpenAI


class AzureChatOpenAI(ChatOpenAI):
 
    def __init__(self, *, api_key: Optional[str] = None, api_version: Optional[str] = "2023-05-15", azure_endpoint: Optional[str] = None, model: str, temperature=0.0):
        _endpoint    = azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        _api_key     = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        _api_version = api_version or os.environ.get("AZURE_OPENAI_API_VERSION")
        self.temperature = temperature
        self.model =  model or os.environ.get("AZURE_OPENAI_API_MODEL")
        self.client = openai.AzureOpenAI(api_key=_api_key, api_version=_api_version, azure_endpoint=_endpoint)
