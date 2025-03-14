import os
import logging
import json
import dirtyjson

from typing import Optional, Union, Any, List, Dict
from openai import OpenAI, AsyncOpenAI

from gwenflow.types import Message, ChatCompletion, ChatCompletionChunk
from gwenflow.llms import ChatBase
from gwenflow.utils import extract_json_str


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

    def _get_model_params(self, tools: list = None, tool_choice: str = None, response_format: str = None) -> Dict[str, Any]:

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

        if tools:
            openai_schema = [tool.openai_schema for tool in tools]
            model_params["tools"] = openai_schema
            model_params["tool_choice"] = tool_choice or "auto"
    
        elif self.tools:
            openai_schema = [tool.openai_schema for tool in self.tools]
            model_params["tools"] = openai_schema or None
            model_params["tool_choice"] = self.tool_choice or "auto"
        
        if response_format:
            model_params["response_format"] = response_format

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

    # def _get_thinking(self, tool_calls):
    #     thinking = []
    #     for tool_call in tool_calls:
    #         if not isinstance(tool_call, dict):
    #             tool_call = tool_call.model_dump()
    #         arguments = json.loads(tool_call["function"]["arguments"])
    #         arguments = ", ".join(arguments.values())
    #         thinking.append(f"""**Calling** { tool_call["function"]["name"].replace("Tool","") } on '{ arguments }'""")
    #     if len(thinking)>0:
    #         return "\n".join(thinking)
    #     return None

    # def invoke(
    #     self,
    #     messages: Union[str, List[Message], List[Dict[str, str]]],
    #     tools: list = None,
    #     tool_choice: str = None,
    #     response_format: str = None,
    # ) -> ModelResponse:

    #     messages_for_model = self._cast_messages(messages)

    #     model_response = ModelResponse()
    #     model_usage = ModelUsage()

    #     while True:
                        
    #         completion = self.get_client().chat.completions.create(
    #             model=self.model,
    #             messages=[self._format_message(m) for m in messages_for_model],
    #             **self._get_model_params(tools=tools, tool_choice=tool_choice, response_format=response_format),
    #         )

    #         model_usage.prompt_tokens     += completion.usage.prompt_tokens
    #         model_usage.completion_tokens += completion.usage.completion_tokens
    #         model_usage.total_tokens      += completion.usage.total_tokens

    #         if not completion.choices[0].message.tool_calls:
    #             model_response.content = completion.choices[0].message.content
    #             model_response.usage = model_usage
    #             break

    #         tool_calls = completion.choices[0].message.tool_calls
    #         model_response.thinking = self._get_thinking(tool_calls=tool_calls)

    #         observations = self.handle_tool_calls(tool_calls=tool_calls)
    #         if len(observations)>0:
    #             messages_for_model.append(Message(**completion.choices[0].message.model_dump()))
    #             messages_for_model.extend(observations)
        
    #     if self.response_format or response_format:
    #         _response_format = response_format or self.response_format
    #         model_response.content = self._parse_response(model_response.content, response_format=_response_format)

    #     return model_response

    # async def ainvoke(
    #     self,
    #     messages: Union[str, List[Message], List[Dict[str, str]]],
    #     tools: list = None,
    #     tool_choice: str = None,
    #     response_format: str = None,
    # ) -> ModelResponse:

    #     messages_for_model = self._cast_messages(messages)

    #     model_response = ModelResponse()
    #     model_usage = ModelUsage()

    #     while True:

    #         completion = await self.get_async_client().chat.completions.create(
    #             model=self.model,
    #             messages=[self._format_message(m) for m in messages_for_model],
    #             **self._get_model_params(tools=tools, tool_choice=tool_choice, response_format=response_format),
    #         )

    #         model_usage.prompt_tokens     += completion.usage.prompt_tokens
    #         model_usage.completion_tokens += completion.usage.completion_tokens
    #         model_usage.total_tokens      += completion.usage.total_tokens

    #         if not completion.choices[0].message.tool_calls:
    #             model_response.content = completion.choices[0].message.content
    #             model_response.usage = model_usage
    #             break

    #         tool_calls = completion.choices[0].message.tool_calls
    #         model_response.thinking = self._get_thinking(tool_calls=tool_calls)

    #         observations = self.handle_tool_calls(tool_calls=tool_calls)
    #         if len(observations)>0:
    #             messages_for_model.append(Message(**completion.choices[0].message.model_dump()))
    #             messages_for_model.extend(observations)

    #     if self.response_format or response_format:
    #         _response_format = response_format or self.response_format
    #         model_response.content = self._parse_response(model_response.content, response_format=_response_format)

    #     return model_response

    # def stream(
    #     self,
    #     messages: Union[str, List[Message], List[Dict[str, str]]],
    #     tools: list = None,
    #     tool_choice: str = None,
    #     response_format: str = None,
    # ):

    #     messages_for_model = self._cast_messages(messages)

    #     model_response = ModelResponse()
    #     model_usage = ModelUsage()

    #     while True:

    #         message = Message(role="assistant", content="", delta="", tool_calls=[])

    #         completion = self.get_client().chat.completions.create(
    #             model=self.model,
    #             messages=[self._format_message(m) for m in messages_for_model],
    #             stream=True,
    #             stream_options={"include_usage": True},
    #             **self._get_model_params(tools=tools, tool_choice=tool_choice, response_format=response_format),
    #         )

    #         for chunk in completion:
    #             model_response.delta = ""
    #             model_response.thinking = ""
    #             if len(chunk.choices)>0:
    #                 if chunk.choices[0].delta.content:
    #                     model_response.delta = chunk.choices[0].delta.content
    #                     message.content     += chunk.choices[0].delta.content
    #                 if chunk.choices[0].delta.tool_calls:
    #                     if chunk.choices[0].delta.tool_calls[0].id:
    #                         message.tool_calls.append(chunk.choices[0].delta.tool_calls[0].model_dump())
    #                     if chunk.choices[0].delta.tool_calls[0].function.arguments:
    #                         current_tool = len(message.tool_calls) - 1
    #                         message.tool_calls[current_tool]["function"]["arguments"] += chunk.choices[0].delta.tool_calls[0].function.arguments

    #             if model_response.delta:
    #                 yield model_response

    #             if chunk.model_dump().get("usage"):
    #                 model_usage.prompt_tokens     += chunk.usage.prompt_tokens
    #                 model_usage.completion_tokens += chunk.usage.completion_tokens
    #                 model_usage.total_tokens      += chunk.usage.total_tokens

    #         if not message.tool_calls:
    #             model_response.content = message.content
    #             model_response.delta = None
    #             model_response.finish_reason = "stop"
    #             model_response.usage = model_usage
    #             break

    #         tool_calls = message.tool_calls
    #         model_response.thinking = self._get_thinking(tool_calls=tool_calls)
 
    #         if model_response.thinking:
    #             yield model_response

    #         observations = self.handle_tool_calls(tool_calls=tool_calls)
    #         if len(observations)>0:
    #             messages_for_model.append(message)
    #             messages_for_model.extend(observations)

    #     if self.response_format or response_format:
    #         _response_format = response_format or self.response_format
    #         model_response.content = self._parse_response(model_response.content, response_format=_response_format)

    #     yield model_response

    # async def astream(
    #     self,
    #     messages: Union[str, List[Message], List[Dict[str, str]]],
    #     tools: list = None,
    #     tool_choice: str = None,
    #     response_format: str = None,
    # ):

    #     messages_for_model = self._cast_messages(messages)

    #     model_response = ModelResponse()
    #     model_usage = ModelUsage()

    #     while True:

    #         message = Message(role="assistant", content="", delta="", tool_calls=[])

    #         completion = self.get_client().chat.completions.create(
    #             model=self.model,
    #             messages=[self._format_message(m) for m in messages_for_model],
    #             stream=True,
    #             stream_options={"include_usage": True},
    #             **self._get_model_params(tools=tools, tool_choice=tool_choice, response_format=response_format),
    #         )

    #         async for chunk in completion:
    #             model_response.delta = ""
    #             model_response.thinking = ""
    #             if len(chunk.choices)>0:
    #                 if chunk.choices[0].delta.content:
    #                     model_response.delta = chunk.choices[0].delta.content
    #                     message.content     += chunk.choices[0].delta.content
    #                 if chunk.choices[0].delta.tool_calls:
    #                     if chunk.choices[0].delta.tool_calls[0].id:
    #                         message.tool_calls.append(chunk.choices[0].delta.tool_calls[0].model_dump())
    #                     if chunk.choices[0].delta.tool_calls[0].function.arguments:
    #                         current_tool = len(message.tool_calls) - 1
    #                         message.tool_calls[current_tool]["function"]["arguments"] += chunk.choices[0].delta.tool_calls[0].function.arguments

    #             if model_response.delta:
    #                 yield model_response

    #             if chunk.model_dump().get("usage"):
    #                 model_usage.prompt_tokens     += chunk.usage.prompt_tokens
    #                 model_usage.completion_tokens += chunk.usage.completion_tokens
    #                 model_usage.total_tokens      += chunk.usage.total_tokens

    #         if not message.tool_calls:
    #             model_response.content = message.content
    #             model_response.delta = None
    #             model_response.finish_reason = "stop"
    #             model_response.usage = model_usage
    #             break

    #         tool_calls = message.tool_calls
    #         model_response.thinking = self._get_thinking(tool_calls=tool_calls)

    #         if model_response.thinking:
    #             yield model_response

    #         observations = await self.ahandle_tool_calls(tool_calls=tool_calls)
    #         if len(observations)>0:
    #             messages_for_model.append(message)
    #             messages_for_model.extend(observations)

    #     if self.response_format or response_format:
    #         _response_format = response_format or self.response_format
    #         model_response.content = self._parse_response(model_response.content, response_format=_response_format)

    #     yield model_response

    def invoke(
        self,
        messages: Union[str, List[Message], List[Dict[str, str]]],
        tools: list = None,
        tool_choice: str = None,
        response_format: str = None,
    ) -> ChatCompletion:
        messages_for_model = self._cast_messages(messages)
                    
        completion = self.get_client().chat.completions.create(
            model=self.model,
            messages=[self._format_message(m) for m in messages_for_model],
            **self._get_model_params(tools=tools, tool_choice=tool_choice, response_format=response_format),
        )

        completion = ChatCompletion(**completion.model_dump())

        if self.response_format or response_format:
            _response_format = response_format or self.response_format
            completion.choices[0].message.content = self._parse_response(completion.choices[0].message.content, response_format=_response_format)

        return completion

    async def ainvoke(
        self,
        messages: Union[str, List[Message], List[Dict[str, str]]],
        tools: list = None,
        tool_choice: str = None,
        response_format: str = None,
    ) -> ChatCompletion:
        messages_for_model = self._cast_messages(messages)

        completion = await self.get_async_client().chat.completions.create(
            model=self.model,
            messages=[self._format_message(m) for m in messages_for_model],
            **self._get_model_params(tools=tools, tool_choice=tool_choice, response_format=response_format),
        )

        completion = ChatCompletion(**completion.model_dump())

        if self.response_format or response_format:
            _response_format = response_format or self.response_format
            completion.choices[0].message.content = self._parse_response(completion.choices[0].message.content, response_format=_response_format)

        return completion

    def stream(
        self,
        messages: Union[str, List[Message], List[Dict[str, str]]],
        tools: list = None,
        tool_choice: str = None,
        response_format: str = None,
    ):
        messages_for_model = self._cast_messages(messages)

        completion = self.get_client().chat.completions.create(
            model=self.model,
            messages=[self._format_message(m) for m in messages_for_model],
            stream=True,
            stream_options={"include_usage": True},
            **self._get_model_params(tools=tools, tool_choice=tool_choice, response_format=response_format),
        )

        for chunk in completion:
            chunk = ChatCompletionChunk(**chunk.model_dump())
            yield chunk

        if self.response_format or response_format:
            _response_format = response_format or self.response_format
            chunk.choices[0].message.content = self._parse_response(chunk.choices[0].message.content, response_format=_response_format)
            yield chunk

    async def astream(
        self,
        messages: Union[str, List[Message], List[Dict[str, str]]],
        tools: list = None,
        tool_choice: str = None,
        response_format: str = None,
    ):
        messages_for_model = self._cast_messages(messages)

        completion = await self.get_client().chat.completions.create(
            model=self.model,
            messages=[self._format_message(m) for m in messages_for_model],
            stream=True,
            stream_options={"include_usage": True},
            **self._get_model_params(tools=tools, tool_choice=tool_choice, response_format=response_format),
        )

        async for chunk in completion:
            chunk = ChatCompletionChunk(**chunk.model_dump())
            yield chunk

        if self.response_format or response_format:
            _response_format = response_format or self.response_format
            chunk.choices[0].message.content = self._parse_response(chunk.choices[0].message.content, response_format=_response_format)
            yield chunk
