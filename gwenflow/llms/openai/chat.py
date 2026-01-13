import json
import os
from typing import Optional, Union, Any, List, Dict, Iterator
from gwenflow.telemetry.base import TelemetryBase
from gwenflow.llms import ChatBase
from gwenflow.types import Message, ItemHelpers
from gwenflow.utils import extract_json_str
from pydantic import Field
from gwenflow.telemetry.openai.openai_instrument import openai_telemetry

from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk


class ChatOpenAI(ChatBase):
 
    model: str = "gpt-4o-mini"

    # model parameters
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    max_completion_tokens: Optional[int] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[Dict[int, float]] = None
    response_format: Optional[Dict[str, Any]] = None
    seed: Optional[int] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None

    # clients
    client: Optional[OpenAI] = None
    async_client: Optional[AsyncOpenAI] = None

    # client parameters
    api_key: Optional[str] = None
    organization: Optional[str] = None
    base_url: Optional[str] = None
    timeout: Optional[Union[float, int]] = None
    max_retries: Optional[int] = None

    # telemetry
    service_name: str = Field(default="gwenflow-service")
    provider: Optional[str] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        telemetry_config = TelemetryBase(service_name=self.service_name)
        self.provider = telemetry_config.setup_telemetry()

        openai_telemetry.instrument()

    def _get_client_params(self) -> Dict[str, Any]:

        api_key = self.api_key
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError(
                "The api_key client option must be set either by passing api_key to the client or by setting the OPENAI_API_KEY environment variable"
            )

        organization = self.organization
        if organization is None:
            organization = os.environ.get('OPENAI_ORG_ID')

        client_params = {
            "api_key": api_key,
            "organization": organization,
            "base_url": self.base_url,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }

        client_params = {k: v for k, v in client_params.items() if v is not None}

        return client_params

    @property
    def _model_params(self) -> Dict[str, Any]:

        model_params = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "n": self.n,
            "stop": self.stop,
            "max_tokens": self.max_tokens or self.max_completion_tokens,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "logit_bias": self.logit_bias,
            "response_format": self.response_format,
            "seed": self.seed,
            "logprobs": self.logprobs,
            "top_logprobs": self.top_logprobs,
        }

        if self.tools and self.tool_type == "fncall":
            model_params["tools"] = [tool.to_openai() for tool in self.tools]
            model_params["tool_choice"] = self.tool_choice or "auto"
        
        model_params = {k: v for k, v in model_params.items() if v is not None}

        return model_params
    
    def get_client(self) -> OpenAI:
        if self.client:
            return self.client
        client_params = self._get_client_params()
        self.client = OpenAI(**client_params)
        return self.client

    def get_async_client(self) -> AsyncOpenAI:
        if self.client:
            return self.client
        client_params = self._get_client_params()
        self.async_client = AsyncOpenAI(**client_params)
        return self.async_client

    def _parse_response(self, response: str, response_format: dict = None) -> str:
        """Process the response."""

        if response_format.get("type") == "json_object":
            try:
                json_str = extract_json_str(response)
                # text_response = dirtyjson.loads(json_str)
                text_response = json.loads(json_str)
                return text_response
            except:
                pass

        return response

    def _format_message(self, message: Message) -> Dict[str, Any]:
        """Format a message into the format expected by OpenAI."""
        return message.to_openai()
    
    def invoke(self, input: Union[str, List[Message], List[Dict[str, str]]]) -> ChatCompletion:
        try:
            messages_for_model = ItemHelpers.input_to_message_list(input)
            completion = self.get_client().chat.completions.create(
                model=self.model,
                messages=[self._format_message(m) for m in messages_for_model],
                **self._model_params,
            )
        except Exception as e:
            raise RuntimeError(f"Error in calling openai API: {e}")

        if self.response_format:
            completion.choices[0].message.content = self._parse_response(completion.choices[0].message.content, response_format=self.response_format)

        return completion

    async def ainvoke(self, input: Union[str, List[Message], List[Dict[str, str]]]) -> ChatCompletion:
        try:
            messages_for_model = ItemHelpers.input_to_message_list(input)
            completion = await self.get_async_client().chat.completions.create(
                model=self.model,
                messages=[self._format_message(m) for m in messages_for_model],
                **self._model_params,
            )
        except Exception as e:
            raise RuntimeError(f"Error in calling openai API: {e}")

        if self.response_format:
            completion.choices[0].message.content = self._parse_response(completion.choices[0].message.content, response_format=self.response_format)

        return completion

    def stream(self, input: Union[str, List[Message], List[Dict[str, str]]]) -> Iterator[ChatCompletionChunk]:
        try:
            messages_for_model = ItemHelpers.input_to_message_list(input)
            yield from self.get_client().chat.completions.create(
                model=self.model,
                messages=[self._format_message(m) for m in messages_for_model],
                stream=True,
                stream_options={"include_usage": True},
                **self._model_params,
            )
        except Exception as e:
            raise RuntimeError(f"Error in calling openai API: {e}")

    async def astream(self, input: Union[str, List[Message], List[Dict[str, str]]]) -> Any:
        try:
            messages_for_model = ItemHelpers.input_to_message_list(input)
            completion = await self.get_async_client().chat.completions.create(
                model=self.model,
                messages=[self._format_message(m) for m in messages_for_model],
                stream=True,
                stream_options={"include_usage": True},
                **self._model_params,
            )
            async for chunk in completion:
                yield chunk
        except Exception as e:
            raise RuntimeError(f"Error in calling openai API: {e}")
