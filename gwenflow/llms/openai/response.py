import json
import os
from typing import Any, AsyncIterator, Dict, Iterator, List, Literal, Optional, Union

from openai import AsyncOpenAI, OpenAI

from gwenflow.llms.base import ChatBase
from gwenflow.types import ItemHelpers, Message
from gwenflow.types.responses import (
    Response,
    ResponseContentDeltaEvent,
    ResponseContentEvent,
    ResponseEvent,
    ResponseEventRoot,
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
    text_format: Optional[Any] = None # TODO change "Any" later to available output type
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
        return {k: v for k, v in client_params.items() if v is not None}

    @property
    def _model_params(self) -> Dict[str, Any]:
        model_params = {
            "background": self.background,
            "max_output_tokens": self.max_output_tokens,
            "max_tool_calls": self.max_tool_calls,
            "prompt_cache_key": self.prompt_cache_key,
            "prompt_cache_retention": self.prompt_cache_retention,
            "temperature": self.temperature,
            "text.format": self.text_format,
            "top_logprobs": self.top_logprobs,
            "top_p": self.top_p,
        }
        if self.get_reasoning_model():
            model_params["reasoning"] = {
                "effort": self.reasoning_effort,
                "summary": self.reasoning_summary
            }

        if self.tools and self.tool_type == "base":
            model_params["tools"] = [tool.to_openai_response() for tool in self.tools]
            model_params["tool_choice"] = self.tool_choice or "auto"

        return {k: v for k, v in model_params.items() if v is not None}

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
        if text_format.get("type") == "json_object":
            try:
                json_str = extract_json_str(response)
                text_response = json.loads(json_str)
                return text_response
            except Exception:
                pass
        return response

    def _format_message(self, message: Message) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        if hasattr(message, "to_openai_response"):
            return message.to_openai_response()
        return message.to_openai_chat_completion()

    def _prepare_input_list(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Helper to flatten the message list for the Responses API. Help also for handling multiple tool calls."""
        api_input_list = []
        for m in messages:
            formatted = self._format_message(m)
            if isinstance(formatted, list):
                api_input_list.extend(formatted)
            else:
                api_input_list.append(formatted)
        return api_input_list

    def _handle_max_output_limit_issue(self, response: Response):
        reason = getattr(response.incomplete_details, "reason", None)
        if response.status == "incomplete" and reason == "max_output_tokens":
            print("Ran out of tokens")
            if response.get_text():
                print("Partial output:", response.get_text())
            else:
                print("Ran out of tokens during generating response")

    def _invoke(self, input: Union[str, List[Message], List[Dict[str, str]]]) -> Response:
        try:
            messages_for_model = ItemHelpers.input_to_message_list(input)

            api_input_list = self._prepare_input_list(messages_for_model)

            raw_response = self.get_client().responses.create(
                model=self.model,
                input=api_input_list,
                **self._model_params,
            )

            response = Response.model_validate(raw_response.model_dump())
        except Exception as e:
            raise RuntimeError(f"Error in calling openai API: {e}") from e

        self._handle_max_output_limit_issue(response)

        if not self.text_format:
            return response

        content_raw = response.get_text()
        parsed_content = self._parse_response(content_raw, text_format=self.text_format)

        for item in response.output:
            if getattr(item, "type", None) == "message":
                item.content = parsed_content

        return response

    async def _ainvoke(self, input: Union[str, List[Message], List[Dict[str, str]]]) -> Response:
        try:
            messages_for_model = ItemHelpers.input_to_message_list(input)

            api_input_list = self._prepare_input_list(messages_for_model)

            raw_response = await self.get_async_client().responses.create(
                model=self.model,
                input=api_input_list,
                **self._model_params,
            )

            response = Response.model_validate(raw_response.model_dump())
        except Exception as e:
            raise RuntimeError(f"Error in calling openai API: {e}") from e

        self._handle_max_output_limit_issue(response)

        if not self.text_format:
            return response

        content_raw = response.get_text()
        parsed_content = self._parse_response(content_raw, text_format=self.text_format)

        for item in response.output:
            if getattr(item, "type", None) == "message":
                item.content = parsed_content

        return response

    def _stream(self, input: Union[str, List[Message], List[Dict[str, str]]]) -> Iterator[ResponseEventRoot]:
        try:
            messages_for_model = ItemHelpers.input_to_message_list(input)
            api_input_list = self._prepare_input_list(messages_for_model)

            client = self.get_client()

            with client.responses.create(
                model=self.model,
                input=api_input_list,
                stream=True,
                **self._model_params,
            ) as raw_stream:
                for raw_event in raw_stream:
                    event_obj = ResponseEventRoot.model_validate(raw_event.model_dump())
                    event = event_obj.root

                    if isinstance(event, ResponseEvent):
                        if event.type == "response.created":
                            yield event_obj

                        elif event.type == "response.completed":
                            self._handle_max_output_limit_issue(event.response)
                            yield event_obj
                            break

                    elif isinstance(event, (ResponseReasoningEvent, ResponseReasoningDeltaEvent)):
                        if self.show_reasoning:
                            yield event_obj

                    elif isinstance(event, (ResponseContentEvent, ResponseContentDeltaEvent, ResponseToolCallEvent)):
                        yield event_obj

        except Exception as e:
            raise RuntimeError(f"Error OpenAI during OpenAI stream : {e}") from e

    async def _astream(self, input: Union[str, List[Message]]) -> AsyncIterator[ResponseEventRoot]:
        try:
            messages_for_model = ItemHelpers.input_to_message_list(input)
            api_input_list = self._prepare_input_list(messages_for_model)

            client = self.get_async_client()

            async with await client.responses.create(
                model=self.model,
                input=api_input_list,
                stream=True,
                **self._model_params,
                ) as raw_stream:
                async for raw_event in raw_stream:
                    event_obj = ResponseEventRoot.model_validate(raw_event.model_dump())
                    event = event_obj.root

                    if isinstance(event, ResponseEvent):
                        if event.type == "response.created":
                            yield event_obj

                        elif event.type == "response.completed":
                            self._handle_max_output_limit_issue(event.response)
                            yield event_obj
                            break

                    elif isinstance(event, (ResponseReasoningEvent, ResponseReasoningDeltaEvent)):
                        if self.show_reasoning:
                            yield event_obj

                    elif isinstance(event, (ResponseContentEvent, ResponseContentDeltaEvent, ResponseToolCallEvent)):
                        yield event_obj

        except Exception as e:
            raise RuntimeError(f"Error OpenAI during OpenAI stream : {e}") from e
