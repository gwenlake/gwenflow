import json
import os
from typing import Any, AsyncIterator, Dict, Iterator, List, Literal, Optional, Union

from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from pydantic import Field

from gwenflow.llms.base import ChatBase
from gwenflow.telemetry.base import TelemetryBase
from gwenflow.telemetry.openai.openai_instrument import openai_telemetry
from gwenflow.types import ItemHelpers, Message
from gwenflow.utils import extract_json_str


class ResponseOpenAI(ChatBase):
    model: str = "gpt-5-mini"

    # model parameters
    background: Optional[bool] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_output_tokens: Optional[int] = None
    max_tool_calls: Optional[int] = None
    prompt_cache_key: Optional[bool] = None
    prompt_cache_retention: Optional[bool] = None
    text_format: Optional[Any] = None
    top_logprobs: Optional[int] = None
    reasoning_effort: Optional[Literal['low', 'medium', 'high']] = None
    reasoning_summary: Optional[Literal['auto', 'concise', 'detailed']] = None #Use only auto to make sure it is compatible with reasoning models
    show_reasoning: bool = False

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
            organization = os.environ.get("OPENAI_ORG_ID")

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
            "background": self.background,
            "max_output_tokens": self.max_output_tokens,
            "max_tool_calls": self.max_tool_calls,
            "prompt_cache_key": self.prompt_cache_key,
            "prompt_cache_retention": self.prompt_cache_retention,
            "temperature": self.temperature,
            "texte.format": self.text_format,
            "top_logprobs": self.top_logprobs,
            "top_p": self.top_p,
        }
        if self.get_reasoning_model():
            model_params["reasoning"] = {
                "effort": self.reasoning_effort,
                "summary": self.reasoning_summary
            }

        if self.tools and self.tool_type == "base":
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

    def _parse_response(self, response: str, text_format: dict = None) -> str:
        """Process the response."""
        if text_format.get("type") == "json_object":
            try:
                json_str = extract_json_str(response)
                # text_response = dirtyjson.loads(json_str)
                text_response = json.loads(json_str)
                return text_response
            except Exception:
                pass

        return response

    def _format_message(self, message: Message) -> Dict[str, Any]:
        """Format a message into the format expected by OpenAI."""
        return message.to_openai()

    def invoke(self, input: Union[str, List[Message], List[Dict[str, str]]]) -> ChatCompletion:
        try:
            messages_for_model = ItemHelpers.input_to_message_list(input)
            responses = self.get_client().responses.create(
                model=self.model,
                input=[self._format_message(m) for m in messages_for_model],
                **self._model_params,
            )
        except Exception as e:
            raise RuntimeError(f"Error in calling openai API: {e}") from e

        if self.text_format:
            responses.choices[0].message.content = self._parse_response(
                responses.choices[0].message.content, text_format=self.text_format
            )

        return responses

    async def ainvoke(self, input: Union[str, List[Message], List[Dict[str, str]]]) -> ChatCompletion:
        try:
            messages_for_model = ItemHelpers.input_to_message_list(input)
            responses = await self.get_async_client().responses.create(
                model=self.model,
                input=[self._format_message(m) for m in messages_for_model],
                **self._model_params,
            )
        except Exception as e:
            raise RuntimeError(f"Error in calling openai API: {e}") from e

        if self.text_format:
            responses.choices[0].message.content = self._parse_response(
                responses.choices[0].message.content, text_format=self.text_format
            )

        return responses

    def stream(self, input: Union[str, List[Message], List[Dict[str, str]]]) -> Iterator[ChatCompletionChunk]:
        try:
            messages_for_model = ItemHelpers.input_to_message_list(input)
            stream = self.get_client().responses.create(
                model=self.model,
                input=[self._format_message(m) for m in messages_for_model],
                stream=True,
                **self._model_params,
            )

            for event in stream:
                if hasattr(event, 'delta') and event.type == 'response.reasoning_summary_text.delta' and self.show_reasoning:
                    yield {'content': event.delta,
                       'type': 'reasoning',
                       'sequence_number': event.sequence_number}

                elif hasattr(event, 'delta') and event.type == 'response.output_text.delta':
                    yield {'content': event.delta,
                           'type': 'output_text',
                           'sequence_number': event.sequence_number}

        except Exception as e:
            raise RuntimeError(f"Error in calling openai API: {e}") from e

    async def astream(self, input: Union[str, List[Message]]) -> AsyncIterator[ChatCompletionChunk]:
        try:
            messages_for_model = ItemHelpers.input_to_message_list(input)

            stream = await self.get_async_client().responses.create(
                model=self.model,
                input=[self._format_message(m) for m in messages_for_model],
                stream=True,
                **self._model_params,
            )

            async for event in stream:
                if hasattr(event, 'delta') and event.type == 'response.reasoning_summary_text.delta' and self.show_reasoning:
                    yield {'content': event.delta,
                       'type': 'reasoning',
                       'sequence_number': event.sequence_number}

                elif hasattr(event, 'delta') and event.type == 'response.output_text.delta':
                    yield {'content': event.delta,
                           'type': 'output_text',
                           'sequence_number': event.sequence_number}

        except Exception as e:
            raise RuntimeError(f"Erreur lors du stream OpenAI : {e}") from e
