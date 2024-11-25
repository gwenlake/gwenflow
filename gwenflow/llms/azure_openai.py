import os
from typing import Optional, Union, List, Dict, Any

import openai

from gwenflow.llms.openai import ChatOpenAI


class ChatAzureOpenAI(ChatOpenAI):
 
    def __init__(
        self,
        *,
        model: str,
        timeout: Optional[Union[float, int]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        n: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        max_completion_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        logit_bias: Optional[Dict[int, float]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        api_version: Optional[str] = None,
        api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        azure_deployment: Optional[str] = None,
        **kwargs,
        ):
        
        # init variables before super().__init__
        _api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        _azure_deployment = azure_deployment or os.environ.get("AZURE_OPENAI_DEPLOYMENT")
        _azure_endpoint = azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        _api_version = api_version or os.environ.get("AZURE_OPENAI_API_VERSION")

        super().__init__(
            model = model,
            timeout = timeout,
            temperature = temperature,
            top_p = top_p,
            n = n,
            stop = stop,
            max_completion_tokens = max_completion_tokens,
            max_tokens = max_tokens,
            presence_penalty = presence_penalty,
            frequency_penalty = frequency_penalty,
            logit_bias = logit_bias,
            response_format = response_format,
            seed = seed,
            logprobs = logprobs,
            top_logprobs = top_logprobs,
            api_key = _api_key,
            **kwargs,
        )


        self.client = openai.AzureOpenAI(
            azure_endpoint=_azure_endpoint,
            azure_deployment=_azure_deployment,
            api_version=_api_version,
            api_key=_api_key,
        )
