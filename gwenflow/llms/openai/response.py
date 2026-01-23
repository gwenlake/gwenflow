import json
import os
from typing import Any, AsyncIterator, Dict, Iterator, List, Literal, Optional, Union

from openai import AsyncOpenAI, OpenAI
from pydantic import Field

from gwenflow.llms.base import ChatBase
from gwenflow.telemetry.base import TelemetryBase
from gwenflow.telemetry.openai.openai_instrument import openai_telemetry
from gwenflow.types import ItemHelpers, Message
from gwenflow.types.responses import (
    Response,
    ResponseContentDeltaEvent,
    ResponseContentEvent,
    ResponseEvent,
    ResponseReasoningDeltaEvent,
    ResponseReasoningEvent,
    ResponseToolCallEvent,
)
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
    text_format: Optional[Any] = None # TODO change later to available output type
    top_logprobs: Optional[int] = None
    reasoning_effort: Optional[Literal['low', 'medium', 'high']] = None
    reasoning_summary: Optional[Literal['auto', 'concise', 'detailed']] = None #Use only auto to make sure it is compatible with all reasoning models for now
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

    # telemetry #TODO move this elsewhere
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
            model_params["tools"] = [tool.to_openai_new() for tool in self.tools]
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

    def invoke(self, input: Union[str, List[Message], List[Dict[str, str]]]) -> Response:
        try:
            messages_for_model = ItemHelpers.input_to_message_list(input)
            raw_response = self.get_client().responses.create(
                model=self.model,
                input=[self._format_message(m) for m in messages_for_model],
                **self._model_params,
            )

            response = Response.model_validate(raw_response.model_dump())
        except Exception as e:
            raise RuntimeError(f"Error in calling openai API: {e}") from e

        if self.text_format:
            content_output = response.get_text()
            content = self._parse_response(
                content_output, text_format=self.text_format
            )
            for item in response.output:
                if item.type == "message" and item.content:
                    item.content[0].text = content
        return response

    async def ainvoke(self, input: Union[str, List[Message], List[Dict[str, str]]]) -> Response:
        try:
            messages_for_model = ItemHelpers.input_to_message_list(input)
            raw_response = await self.get_async_client().responses.create(
                model=self.model,
                input=[self._format_message(m) for m in messages_for_model],
                **self._model_params,
            )

            response = Response.model_validate(raw_response.model_dump())
        except Exception as e:
            raise RuntimeError(f"Error in calling openai API: {e}") from e

        if self.text_format:
            content_output = response.get_text()
            content = self._parse_response(
                content_output, text_format=self.text_format
            )
            for item in response.output:
                if item.type == "message" and item.content:
                    item.content[0].text = content
        return response

    def stream(self, input: Union[str, List[Message], List[Dict[str, str]]]) -> Iterator[ResponseEvent]:
        try:
            messages_for_model = ItemHelpers.input_to_message_list(input)
            client = self.get_client()

            with client.responses.create(
                model=self.model,
                input=[self._format_message(m) for m in messages_for_model],
                stream=True,
                **self._model_params,
            ) as raw_stream:
                for raw_event in raw_stream:
                    try:
                        event_obj = ResponseEvent.model_validate(raw_event.model_dump())
                        event = event_obj.root

                        if isinstance(event, (ResponseReasoningEvent, ResponseReasoningDeltaEvent)):
                            if self.show_reasoning:
                                yield event_obj

                        elif isinstance(event, (ResponseContentEvent, ResponseContentDeltaEvent)):
                            yield event_obj

                        elif isinstance(event, ResponseToolCallEvent):
                            yield event_obj

                        if getattr(event, "type", None) == "response.done":
                            break

                    except Exception:
                        continue

        except Exception as e:
            raise RuntimeError(f"Erreur lors du stream OpenAI : {e}") from e

    async def astream(self, input: Union[str, List[Message]]) -> AsyncIterator[ResponseEvent]:
        try:
            messages_for_model = ItemHelpers.input_to_message_list(input)

            client = self.get_async_client()

            async with await client.responses.create(
                model=self.model,
                input=[self._format_message(m) for m in messages_for_model],
                stream=True,
                **self._model_params,
                ) as raw_stream:
                async for raw_event in raw_stream:
                    try:
                        event_obj = ResponseEvent.model_validate(raw_event.model_dump())
                        event = event_obj.root

                        if isinstance(event, (ResponseReasoningEvent, ResponseReasoningDeltaEvent)):
                            if self.show_reasoning:
                                yield event_obj

                        elif isinstance(event, (ResponseContentEvent, ResponseContentDeltaEvent)):
                            yield event_obj

                        elif isinstance(event, ResponseToolCallEvent):
                            yield event_obj

                        if getattr(event, "type", None) == "response.done":
                            break

                    except Exception:
                        continue
        except Exception as e:
            raise RuntimeError(f"Erreur lors du stream OpenAI : {e}") from e
