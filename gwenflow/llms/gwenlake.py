import os
from typing import Optional, Union

import openai

from gwenflow.llms.openai import ChatOpenAI


class ChatGwenlake(ChatOpenAI):
 
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str,
        temperature: Optional[float] = 0.0,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
    ):
        _api_key = api_key or os.environ.get("GWENLAKE_API_KEY")
        if os.environ.get('OPENAI_ORG_ID'):
            openai.organization = os.environ.get('OPENAI_ORG_ID')
        if os.environ.get('GWENLAKE_ORGANIZATION'):
            openai.organization = os.environ.get('GWENLAKE_ORGANIZATION')

        self.client = openai.OpenAI(api_key=_api_key, base_url="https://api.gwenlake.com/v1")
        
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
