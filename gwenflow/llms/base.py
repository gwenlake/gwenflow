from abc import ABC, abstractmethod
from typing import Optional, Union, Mapping, Any, List, Dict, Iterator, AsyncIterator


class ChatBase(ABC):
 
    def _get_system(self, messages: list[dict]):
        for message in messages:
            if message["role"] == "system":
                return message
        return None

    def _get_messages(self, messages: list[dict]):
        filtered_messages = []
        for message in messages:
            if message["role"] != "system":
                filtered_messages.append(message)
        return filtered_messages

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
        **kwargs,
    ):
        self.config = {
            "model": model,
            "timeout": timeout,
            "temperature": temperature,
            "top_p": top_p,
            "n": n,
            "stop": stop,
            "max_tokens": max_tokens or max_completion_tokens,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "logit_bias": logit_bias,
            "response_format": response_format,
            "seed": seed,
            "logprobs": logprobs,
            "top_logprobs": top_logprobs,
            "stream": False,
            **kwargs,
        }

        # Remove None values to avoid passing unnecessary parameters
        self.config = {k: v for k, v in self.config.items() if v is not None}
 
    @abstractmethod
    def invoke(self, messages):
        """
        Generate a response based on the given messages.

        Args:
            messages (list): List of message dicts containing 'role' and 'content'.
            response_format (str or object, optional): Format of the response. Defaults to "text".
            tools (list, optional): List of tools that the model can call. Defaults to None.
            tool_choice (str, optional): Tool choice method. Defaults to "auto".

        Returns:
            str: The generated response.
        """
        pass

    @abstractmethod
    def stream(self, messages):
        """
        Generate a streamed response based on the given messages.

        Args:
            messages (list): List of message dicts containing 'role' and 'content'.
            response_format (str or object, optional): Format of the response. Defaults to "text".
            tools (list, optional): List of tools that the model can call. Defaults to None.
            tool_choice (str, optional): Tool choice method. Defaults to "auto".

        Returns:
            str: The generated response.
        """
        pass
