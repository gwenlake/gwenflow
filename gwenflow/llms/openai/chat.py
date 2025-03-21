import dirtyjson
import json
import logging
import os
from typing import Optional, Union, Any, List, Dict, Iterator

from gwenflow.llms import ChatBase
from gwenflow.llms.response import ModelResponse
from gwenflow.types import Message, ChatCompletion, ChatCompletionChunk
from gwenflow.utils import extract_json_str
from openai import OpenAI, AsyncOpenAI

logger = logging.getLogger(__name__)


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

        if self.tools:
            openai_schema = [tool.openai_schema for tool in self.tools]
            model_params["tools"] = openai_schema or None
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

        message_dict: Dict[str, Any] = {
            "role": message.role,
            "content": message.content,
            "name": message.name,
            "tool_call_id": message.tool_call_id,
            "tool_calls": message.tool_calls,
        }
        message_dict = {k: v for k, v in message_dict.items() if v is not None}

        # OpenAI expects the tool_calls to be None if empty, not an empty list
        if message.tool_calls is not None and len(message.tool_calls) == 0:
            message_dict["tool_calls"] = None

        # Manually add the content field even if it is None
        if message.content is None:
            message_dict["content"] = None

        return message_dict

    def _get_thinking(self, tool_calls) -> list:
        thinking = []
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                tool_call = tool_call.model_dump()
            arguments = json.loads(tool_call["function"]["arguments"])
            arguments = ", ".join(arguments.values())
            thinking.append(f"""**Calling** { tool_call["function"]["name"].replace("Tool","") } on '{ arguments }'""")
        if len(thinking)>0:
            # return "\n".join(thinking)
            return thinking
        return []
    
    def invoke(self, messages: Union[str, List[Message], List[Dict[str, str]]]) -> ChatCompletion:
        messages_for_model = self._cast_messages(messages)

        try:
                    
            completion = self.get_client().chat.completions.create(
                model=self.model,
                messages=[self._format_message(m) for m in messages_for_model],
                **self._model_params,
            )

        except Exception as e:
            raise RuntimeError(f"Error in calling openai API: {e}")

        completion = ChatCompletion(**completion.model_dump())

        if self.response_format:
            completion.choices[0].message.content = self._parse_response(completion.choices[0].message.content, response_format=self.response_format)

        return completion

    async def ainvoke(self, messages: Union[str, List[Message], List[Dict[str, str]]]) -> ChatCompletion:
        messages_for_model = self._cast_messages(messages)

        try:

            completion = await self.get_async_client().chat.completions.create(
                model=self.model,
                messages=[self._format_message(m) for m in messages_for_model],
                **self._model_params,
            )
        except Exception as e:
            raise RuntimeError(f"Error in calling openai API: {e}")

        completion = ChatCompletion(**completion.model_dump())

        if self.response_format:
            completion.choices[0].message.content = self._parse_response(completion.choices[0].message.content, response_format=self.response_format)

        return completion

    def stream(self, messages: Union[str, List[Message], List[Dict[str, str]]]) -> Iterator[ChatCompletionChunk]:
        messages_for_model = self._cast_messages(messages)

        try:
            completion = self.get_client().chat.completions.create(
                model=self.model,
                messages=[self._format_message(m) for m in messages_for_model],
                stream=True,
                stream_options={"include_usage": True},
                **self._model_params,
            )
        except Exception as e:
            raise RuntimeError(f"Error in calling openai API: {e}")

        for chunk in completion:
            chunk = ChatCompletionChunk(**chunk.model_dump())
            yield chunk

    async def astream(self, messages: Union[str, List[Message], List[Dict[str, str]]]) ->  Any:
        messages_for_model = self._cast_messages(messages)

        try:

            completion = await self.get_client().chat.completions.create(
                model=self.model,
                messages=[self._format_message(m) for m in messages_for_model],
                stream=True,
                stream_options={"include_usage": True},
                **self._model_params,
            )
        except Exception as e:
            raise RuntimeError(f"Error in calling openai API: {e}")

        async for chunk in completion:
            chunk = ChatCompletionChunk(**chunk.model_dump())
            yield chunk

    def response(self, messages: Union[str, List[Message], List[Dict[str, str]]]) -> ModelResponse:
        messages_for_model = self._cast_messages(messages)
        model_response = ModelResponse()

        while True:

            response = self.invoke(messages=messages_for_model)
            
            if not response.choices[0].message.tool_calls:
                model_response.content = response.choices[0].message.content
                break

            model_response.thinking.extend(self._get_thinking(response.choices[0].message.tool_calls))

            tool_calls = response.choices[0].message.tool_calls
            if tool_calls and self.tools:
                tool_messages = self.handle_tool_calls(tool_calls=tool_calls)
                if len(tool_messages)>0:
                    messages_for_model.append(response.choices[0].message.model_dump())
                    messages_for_model.extend(tool_messages)
        
        return model_response

    async def aresponse(self, messages: Union[str, List[Message], List[Dict[str, str]]]) -> ModelResponse:
        messages_for_model = self._cast_messages(messages)
        model_response = ModelResponse()

        while True:

            response = await self.ainvoke(messages=messages_for_model)

            if not response.choices[0].message.tool_calls:
                model_response.content = response.choices[0].message.content
                break

            model_response.thinking.extend(self._get_thinking(response.choices[0].message.tool_calls))

            tool_calls = response.choices[0].message.tool_calls
            if tool_calls and self.tools:
                tool_messages = await self.ahandle_tool_calls(tool_calls=tool_calls)
                if len(tool_messages)>0:
                    messages_for_model.append(response.choices[0].message.model_dump())
                    messages_for_model.extend(tool_messages)

        return model_response

    def response_stream(self, messages: Union[str, List[Message], List[Dict[str, str]]]) -> Iterator[ModelResponse]:
        messages_for_model = self._cast_messages(messages)
        model_response = ModelResponse()

        while True:

            message = Message(role="assistant", content="", delta="", tool_calls=[])

            for chunk in self.stream(messages=messages_for_model):
                chunk = ChatCompletionChunk(**chunk.model_dump())
                if len(chunk.choices) > 0:
                    if chunk.choices[0].delta.content:
                        message.content += chunk.choices[0].delta.content
                        model_response.content = chunk.choices[0].delta.content
                        yield model_response
                    elif chunk.choices[0].delta.tool_calls:
                        if chunk.choices[0].delta.tool_calls[0].id:
                            message.tool_calls.append(chunk.choices[0].delta.tool_calls[0].model_dump())
                        if chunk.choices[0].delta.tool_calls[0].function.arguments:
                            current_tool = len(message.tool_calls) - 1
                            message.tool_calls[current_tool]["function"]["arguments"] += chunk.choices[0].delta.tool_calls[0].function.arguments

            if not message.tool_calls:
                model_response.content = None
                model_response.finish_reason = "stop"
                break

            model_response.thinking = self._get_thinking(message.tool_calls)
            if model_response.thinking:
                yield model_response

            tool_calls = message.tool_calls
            if tool_calls and self.tools:
                tool_messages = self.handle_tool_calls(tool_calls=tool_calls)
                if len(tool_messages)>0:
                    messages_for_model.append(message)
                    messages_for_model.extend(tool_messages)

        yield model_response

    async def aresponse_stream(self, messages: Union[str, List[Message], List[Dict[str, str]]]) -> Any:
        messages_for_model = self._cast_messages(messages)
        model_response = ModelResponse()

        while True:

            model_response.thinking = []

            message = Message(role="assistant", content="", delta="", tool_calls=[])

            for chunk in await self.astream(messages=messages_for_model):
                chunk = ChatCompletionChunk(**chunk.model_dump())
                if len(chunk.choices) > 0:
                    if chunk.choices[0].delta.content:
                        message.content += chunk.choices[0].delta.content
                        model_response.content = chunk.choices[0].delta.content
                        yield model_response
                    elif chunk.choices[0].delta.tool_calls:
                        if chunk.choices[0].delta.tool_calls[0].id:
                            message.tool_calls.append(chunk.choices[0].delta.tool_calls[0].model_dump())
                        if chunk.choices[0].delta.tool_calls[0].function.arguments:
                            current_tool = len(message.tool_calls) - 1
                            message.tool_calls[current_tool]["function"]["arguments"] += chunk.choices[0].delta.tool_calls[0].function.arguments

            if not message.tool_calls:
                model_response.content = None
                model_response.finish_reason = "stop"
                break

            model_response.thinking = self._get_thinking(message.tool_calls)
            if model_response.thinking:
                yield model_response

            tool_calls = message.tool_calls
            if tool_calls and self.tools:
                tool_messages = await self.ahandle_tool_calls(tool_calls=tool_calls)
                if len(tool_messages)>0:
                    messages_for_model.append(message)
                    messages_for_model.extend(tool_messages)

        yield model_response
