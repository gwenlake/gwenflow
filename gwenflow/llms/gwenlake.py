import os
from typing import Optional, Union

import openai

from gwenflow.llms.openai import ChatOpenAI


class ChatGwenlake(ChatOpenAI):
 
    def __init__(self, *, api_key: Optional[str] = None, endpoint: Optional[str] = None, model: str, temperature=0.0):
        _api_key = api_key or os.environ.get("GWENLAKE_API_KEY")
        _base_url = endpoint or os.environ.get("GWENLAKE_ENDPOINT")
        self.temperature = temperature
        self.model =  model
        self.client = openai.OpenAI(api_key=_api_key, base_url=_base_url)
