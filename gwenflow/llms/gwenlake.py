import os
from typing import Optional, Union

import openai

from gwenflow.llms.openai import OpenAI


class Gwenlake(OpenAI):
 
    def __init__(self, *, api_key: Optional[str] = None, model: str, temperature=0.0):
        _api_key = api_key or os.environ.get("GWENLAKE_API_KEY")
        if os.environ.get('OPENAI_ORG_ID'):
            openai.organization = os.environ.get('OPENAI_ORG_ID')
        if os.environ.get('GWENLAKE_ORGANIZATION'):
            openai.organization = os.environ.get('GWENLAKE_ORGANIZATION')
        self.temperature = temperature
        self.model =  model
        self.client = openai.OpenAI(api_key=_api_key, base_url="https://api.gwenlake.com/v1")
