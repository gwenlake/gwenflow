
import uuid
import json
from typing import List, Callable, Union, Optional, Any, Dict, Iterator, Literal, Sequence, overload, Type
from collections import defaultdict
from pydantic import BaseModel, model_validator, field_validator, Field
from datetime import datetime

from gwenflow.llms import ChatOpenAI
from gwenflow.types import ChatCompletionMessage, ChatCompletionMessageToolCall
from gwenflow.tools import BaseTool
from gwenflow.memory import ChatMemoryBuffer
from gwenflow.agents.run import RunResponse
from gwenflow.agents.utils import merge_chunk
from gwenflow.utils import logger


MAX_TURNS = 10


class Result(BaseModel):
    """Encapsulates the possible return values for an agent function."""
    value: str = ""
    agent: Optional[Any] = None
    context_variables: dict = {}


class Agent(BaseModel):

    # --- Agent Settings
    id: Optional[str] = Field(None, validate_default=True)
    name: str

    # --- Settings for system message
    description: Optional[str] = "You are a helpful AI assistant."
    task: Optional[str] = None
    instructions: Optional[Union[str, List[str]]] = []
    add_datetime_to_instructions: bool = True
    markdown: bool = False
    scrape_links: bool = True

    response_model: Optional[str] = None
 
    # --- Agent Model and Tools
    llm: Optional[Any] = Field(None, validate_default=True)
    tools: List[BaseTool] = []
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None

    # --- Context and Memory
    context_vars: Optional[List[str]] = []
    memory: Optional[ChatMemoryBuffer] = None
    keep_history: bool = False
    metadata: Optional[Dict[str, Any]] = None

    # --- Team of agents
    team: Optional[List["Agent"]] = None


    @field_validator("id", mode="before")
    def set_id(cls, v: Optional[str]) -> str:
        id = v or str(uuid.uuid4())
        return id

    @field_validator("instructions", mode="before")
    def set_instructions(cls, v: Optional[Union[List, str]]) -> str:
        if isinstance(v, str):
            instructions = [v]
            return instructions
        return v

    @field_validator("llm", mode="before")
    def set_llm(cls, v: Optional[Any]) -> str:
        llm = v or ChatOpenAI(model="gpt-4o-mini")
        return llm

    @model_validator(mode="after")
    def model_valid(self) -> Any:
        if self.memory is None and self.llm is not None:
             token_limit = self.llm.get_context_window_size()
             self.memory = ChatMemoryBuffer(token_limit=token_limit)
        return self
    
    def get_system_message(self, context: Optional[Any] = None):
        """Return the system message for the Agent."""

        system_message_lines = []

        if self.description is not None:
            system_message_lines.append(f"{self.description}\n")

        if self.name is not None:
            system_message_lines.append(f"Your name is: {self.name}.\n")

        if self.task is not None:
            system_message_lines.append(f"Your task is: {self.task}\n")

        # instructions
        instructions = self.instructions
        
        if self.add_datetime_to_instructions:
            instructions.append(f"The current time is { datetime.now() }")

        if self.markdown and self.response_model is None:
            instructions.append("Use markdown to format your answers.")

        if self.scrape_links:
            instructions.append("If you get a list of web links, systematically scrape the content of all the linked websites to extract detailed information about the topic.")

        if self.response_model:
             instructions.append("Use JSON to format your answers.")

        if context is not None:
            instructions.append("Always prefer information from the provided context over your own knowledge.")

        if len(instructions) > 0:
            system_message_lines.append("# Instructions")
            system_message_lines.extend([f"- {instruction}" for instruction in instructions])
            system_message_lines.append("")

        if self.response_model:
            system_message_lines.append("# Provide your output using the following JSON schema:")
            if isinstance(self.response_model, str):
                system_message_lines.append("<json_fields>")
                system_message_lines.append(f"{ self.response_model.strip() }")
                system_message_lines.append("</json_fields>\n\n")

        # final system prompt
        if len(system_message_lines) > 0:
            return dict(role="system", content=("\n".join(system_message_lines)).strip())
        
        return None

    def get_user_message(self, user_prompt: Optional[str] = None, context: Optional[Any] = None):
        """Return the user message for the Agent."""

        prompt = ""

        if context is not None:

            prompt += "\n\nUse the following information from the knowledge base if it helps:\n"
            prompt += "<context>\n"

            if isinstance(context, str):
                prompt += context + "\n"

            elif isinstance(context, dict):
                for key in context.keys():
                    prompt += f"<{key}>\n"
                    prompt += context.get(key) + "\n"
                    prompt += f"</{key}>\n\n"

            prompt += "</context>\n\n"
        
        if user_prompt:
            if isinstance(user_prompt, str):
                prompt += user_prompt
            elif isinstance(user_prompt, dict):
                prompt += user_prompt["content"]
        
        return { "role": "user", "content": prompt }

    
    def get_tools_openai_schema(self, tools: List[BaseTool]):
        return [tool.openai_schema for tool in tools]

    def get_tools_map(self, tools: List[BaseTool]):
        return {tool.name: tool for tool in tools}

    def handle_function_result(self, result) -> Result:
        match result:
            case Result() as result:
                return result

            case Agent() as agent:
                return Result(
                    value=json.dumps({"assistant": self.name}),
                    agent=agent,
                )
            case _:
                try:
                    return Result(value=str(result))
                except Exception as e:
                    error_message = f"Failed to cast response to string: {result}. Make sure agent functions return a string or Result object. Error: {str(e)}"
                    logger.error(error_message)
                    raise TypeError(error_message)

    def handle_tool_calls(
        self,
        tool_calls: List[ChatCompletionMessageToolCall],
        tools: List[BaseTool],
    ) -> RunResponse:
        
        tool_map = self.get_tools_map(self.tools)

        partial_response = RunResponse(messages=[], agent=None)

        for tool_call in tool_calls:

            name = tool_call.function.name

            # handle missing tool case, skip to next tool
            if name not in tool_map:
                logger.debug(f"Tool {name} not found in function map.")
                partial_response.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "tool_name": name,
                        "content": f"Error: Tool {name} not found.",
                    }
                )
                continue

            args = json.loads(tool_call.function.arguments)
            logger.debug(f"Tool call: {name} with arguments {args}")

            tool_result = tool_map[name].run(**args)

            result: Result = self.handle_function_result(tool_result)

            partial_response.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "tool_name": name,
                    "content": result.value,
                }
            )

            if result.agent:
                partial_response.agent = result.agent

        return partial_response

    def invoke(self, messages: list, stream: bool = False) ->  Union[Any, Iterator[Any]]:

        tools = self.get_tools_openai_schema(self.tools)

        params = {
            "messages": messages,
            "tools": tools or None,
            "tool_choice": self.tool_choice,
            "parse_response": False,
        }

        response_format = None
        if self.response_model:
            response_format = {"type": "json_object"}

        if stream:
            return self.llm.stream(**params, response_format=response_format)
        
        return self.llm.invoke(**params, response_format=response_format)


    def _run(
        self,
        user_prompt: Optional[str] = None,
        *,
        context: Optional[Any] = None,
        stream: Optional[bool] = False,
        **kwargs: Any,
    ) ->  Iterator[RunResponse]:

        # prepare messages
        messages_for_model = []
        system_message = self.get_system_message(context=context)
        if system_message:
            messages_for_model.append(system_message)

        if self.keep_history:
            if len(self.memory.get())>0:
                messages_for_model.extend(self.memory.get())

        user_message = self.get_user_message(user_prompt, context=context)
        if user_message:
            messages_for_model.append(user_message)
            if self.memory and self.keep_history:
                self.memory.add_message(user_message)
                
        # global loop
        init_len = len(messages_for_model)
        while len(messages_for_model) - init_len < MAX_TURNS:

            if stream:
                message = {
                    "content": "",
                    "sender": self.name,
                    "role": "assistant",
                    "function_call": None,
                    "tool_calls": defaultdict(
                        lambda: {
                            "function": {"arguments": "", "name": ""},
                            "id": "",
                            "type": "",
                        }
                    ),
                }

                completion = self.invoke(messages=messages_for_model, stream=True)

                for chunk in completion:
                    if len(chunk.choices) > 0:
                        delta = json.loads(chunk.choices[0].delta.json())
                        if delta["role"] == "assistant":
                            delta["sender"] = self.name
                        if delta["content"]:
                            yield delta["content"]
                        delta.pop("role", None)
                        delta.pop("sender", None)
                        merge_chunk(message, delta)

                message["tool_calls"] = list(message.get("tool_calls", {}).values())
                message = ChatCompletionMessage(**message)
            
            else:
                completion = self.invoke(messages=messages_for_model)
                message = completion.choices[0].message
                message.sender = self.name

            # add messages to the current message stack
            message_dict = json.loads(message.model_dump_json())
            messages_for_model.append(message_dict)

            if not message.tool_calls:
                self.memory.add_message(message_dict)
                break

            # handle tool calls and switching agents
            partial_response = self.handle_tool_calls(message.tool_calls, self.tools)
            messages_for_model.extend(partial_response.messages)
            if partial_response.agent:
                return partial_response.agent

        content = messages_for_model[-1]["content"]
        if self.response_model:
            content = json.loads(content)

        yield RunResponse(
            content=content,
            messages=messages_for_model[init_len:],
            agent=self,
            tools=self.tools,
        )


    def run(
        self,
        user_prompt: Optional[str] = None,
        *,
        context: Optional[Any] = None,
        stream: Optional[bool] = False,
        output_file: Optional[str] = None,
        **kwargs: Any,
    ) ->  Union[RunResponse, Iterator[RunResponse]]:


        agent_id = self.name or self.id

        logger.debug("")
        logger.debug("------------------------------------------")
        logger.debug(f"Running Agent: { agent_id }")
        logger.debug("------------------------------------------")
        logger.debug("")

        if stream:
            response = self._run(
                user_prompt=user_prompt,
                context=context,
                stream=True,
                **kwargs,
            )
            return response
    
        else:

            response = self._run(
                user_prompt=user_prompt,
                context=context,
                stream=False,
                **kwargs,
            )
            response = next(response)

            if output_file:
                with open(output_file, "a") as file:

                    name = self.name or self.id

                    file.write("\n")
                    file.write("---\n\n")
                    file.write(f"# Agent: { name }\n")
                    if self.task:
                        file.write(f"{ self.task }\n")
                    file.write("\n")
                    file.write(response.content)

            return response
