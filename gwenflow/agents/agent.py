
import uuid
import json
import inspect
from typing import List, Union, Optional, Any, Dict, Iterator
from collections import defaultdict
from pydantic import BaseModel, model_validator, field_validator, Field, UUID4
from datetime import datetime

from gwenflow.llms import BaseModel, ChatOpenAI
from gwenflow.types import Message, ChatCompletionMessage, ChatCompletionMessageToolCall, ChatCompletionChunk
from gwenflow.tools import BaseTool
from gwenflow.memory import ChatMemoryBuffer
from gwenflow.agents.types import AgentResponse
from gwenflow.agents.prompts import PROMPT_TOOLS, PROMPT_STEPS, PROMPT_GUIDELINES, PROMPT_JSON_SCHEMA, PROMPT_TOOLS_REACT_GUIDELINES, PROMPT_TASK, PROMPT_REASONING_STEPS_TOOLS
from gwenflow.utils import logger
from gwenflow.utils.chunks import merge_chunk
from gwenflow.types import Message


MAX_TURNS = float('inf') #10


class Agent(BaseModel):

    # --- Agent Settings
    id: UUID4 = Field(default_factory=uuid.uuid4, frozen=True)
    name: str = Field(description="Name of the agent")
    role: str = Field(description="Role of the agent")
    description: Optional[str] = Field(default=None, description="Description of the agent")

    # --- Settings for system message
    system_prompt: Optional[str] = None
    instructions: Optional[Union[str, List[str]]] = []
    add_datetime_to_instructions: bool = True
    markdown: bool = False
    response_model: Optional[dict] = None
    is_react: bool = False
    system_prompt_allowed: bool = True
 
    # --- Agent Model and Tools
    llm: Optional[BaseModel] = Field(None, validate_default=True)
    tools: List[BaseTool] = []
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    show_tool_calls: bool = False

    # --- Reasoning models
    reasoning_steps: Optional[str] = None
    reasoning_model: Optional[Any] = Field(None, validate_default=True)

    # --- Task, Context and Memory
    context: Optional[Any] = None
    memory: Optional[ChatMemoryBuffer] = None
    metadata: Optional[Dict[str, Any]] = None
    # knowledge: Optional[Knowledge] = None

    # --- Team of agents
    team: Optional[List["Agent"]] = None


    @field_validator("id", mode="before")
    @classmethod
    def deny_user_set_id(cls, v: Optional[UUID4]) -> None:
        if v:
            raise ValueError("This field is not to be set by the user.")

    @field_validator("instructions", mode="before")
    @classmethod
    def set_instructions(cls, v: Optional[Union[List, str]]) -> str:
        if isinstance(v, str):
            instructions = [v]
            return instructions
        return v

    @field_validator("llm", mode="before")
    @classmethod
    def set_llm(cls, v: Optional[Any]) -> Any:
        llm = v or ChatOpenAI(model="gpt-4o-mini")
        return llm

    @model_validator(mode="after")
    def model_valid(self) -> Any:
        if self.memory is None and self.llm is not None:
             token_limit = self.llm.get_context_window_size()
             self.memory = ChatMemoryBuffer(token_limit=token_limit)
        return self
    
    def get_system_prompt(self) -> str:
        """Return the system message for the Agent."""

        prompt = ""

        if self.system_prompt is not None:
            prompt = self.system_prompt.strip()
            prompt += "\n"
        else:
            prompt += f"You are an AI agent named '{self.name}'."
            if self.role:
                prompt += f" {self.role.strip('.')}."
            if self.description:
                prompt += f" {self.description.strip('.')}."
            prompt += "\n\n"
            if self.add_datetime_to_instructions:
                prompt += f"The current date and time is: { datetime.now() }\n"

        # tools: TODO REMOVE ?
        if self.tools and self.is_react:
            tools = self.get_tools_text_schema()
            prompt += PROMPT_TOOLS.format(tools=tools).strip()
            prompt += PROMPT_TOOLS_REACT_GUIDELINES

        # instructions
        guidelines = []        
        if self.response_model:
            guidelines.append("Use JSON to format your answers.")
        elif self.markdown:
            guidelines.append("Use markdown to format your answers.")
        if self.tools is not None:
            guidelines.append("Only use the tools you are provided.")
        if self.context is not None:
            guidelines.append("Always prefer information from the provided context over your own knowledge.")

        if len(self.instructions) > 0:
            guidelines += self.instructions

        if len(guidelines) > 0:
            prompt += PROMPT_GUIDELINES.format(guidelines="\n".join([f"- {g}" for g in guidelines]))
            prompt += "\n"

        if self.reasoning_steps:
            prompt += PROMPT_STEPS.format(reasoning_steps=self.reasoning_steps)
            prompt += "\n"

        if self.response_model:
            prompts += PROMPT_JSON_SCHEMA.format(json_schema=json.dumps(self.response_model, indent=4))
            prompt += "\n"
        
        return prompt.strip()


    def get_context(self):
        prompt  = "Use the following information if it helps:\n\n"

        if isinstance(self.context, str):
            prompt += "<context>\n"
            prompt += self.context + "\n"
            prompt += "</context>\n\n"

        elif isinstance(self.context, dict):
            for key in self.context.keys():
                prompt += f"<{key}>\n"
                prompt += self.context.get(key) + "\n"
                prompt += f"</{key}>\n\n"

        return prompt

    def get_user_message(self, task: Optional[str] = None) -> Message:
        """Return the user message for the Agent."""

        if not task and not self.context:
            raise ValueError("You need a task or a context (or both) to run the agent!")

        prompt = ""

        if self.context:
            prompt += self.get_context()

        if task:
            prompt += PROMPT_TASK.format(task=task)

        return Message(role="user", content=prompt)

    def get_tools_openai_schema(self):
        return [tool.openai_schema for tool in self.tools]

    def get_tools_text_schema(self) -> str:
        descriptions = []
        for tool in self.tools:
            sig = inspect.signature(tool._run)
            description = f"{tool.name}{sig} - {tool.description}"
            descriptions.append(description)
        return "\n".join(descriptions)

    def get_tool_map(self):
        return {tool.name: tool for tool in self.tools}

    def get_tool_names(self):
        return [tool.name for tool in self.tools]

    def execute_tool_call(self, tool_name: str, arguments: Dict[str, str]) -> Any:
        available_tools = self.get_tool_map()
        if tool_name not in available_tools:
            logger.error(f"Unknown tool {tool_name}, should be instead one of { available_tools.keys() }.")
            return None

        logger.debug(f"Tool call: {tool_name}({arguments})")
        observation = available_tools[tool_name].run(**arguments)

        return observation

    def handle_tool_calls(
        self,
        tool_calls: List[ChatCompletionMessageToolCall],
    ) -> List[Message]:
        
        tool_map = self.get_tool_map()

        messages = []
        for tool_call in tool_calls:

            tool_name = tool_call.function.name

            # handle missing tool case, skip to next tool
            if tool_name not in tool_map.keys():
                logger.warning(f"Unknown tool {tool_name}, should be instead one of { tool_map.keys() }.")
                messages.append(
                    Message(
                        role="tool",
                        tool_call_id=tool_call.id,
                        tool_name=tool_name,
                        content=f"Tool {tool_name} does not exist",
                    )
                )
                continue

            arguments = json.loads(tool_call.function.arguments)
            observation = self.execute_tool_call(tool_name, arguments)
            
            if observation:
                messages.append(
                    Message(
                        role="tool",
                        tool_call_id=tool_call.id,
                        tool_name=tool_name,
                        content=f"Observation: {observation}",
                    )
                )

        return messages

    def invoke(self, messages: list, stream: bool = False) ->  Union[Any, Iterator[Any]]:

        tools = self.get_tools_openai_schema()

        params = {
            "messages": messages,
            "tools": tools or None,
            "tool_choice": self.tool_choice,
        }

        if self.response_model:
            params["response_format"] = {"type": "json_object"}

        if stream:
            return self.llm.stream(**params)
        
        return self.llm.invoke(**params)

    async def ainvoke(self, messages: list, stream: bool = False) ->  Union[Any, Iterator[Any]]:

        tools = self.get_tools_openai_schema()

        params = {
            "messages": messages,
            "tools": tools or None,
            "tool_choice": self.tool_choice,
        }

        if self.response_model:
            params["response_format"] = {"type": "json_object"}

        if stream:
            return self.llm.astream(**params)
        
        return self.llm.ainvoke(**params)
    
    def _run(
        self,
        task: Optional[str] = None,
        stream: Optional[bool] = False,
        chat_history: Optional[List] = None,
    ) ->  Iterator[AgentResponse]:

        # add reasoning steps
        if self.reasoning_model:
            tools = self.get_tools_text_schema()
            completion = self.invoke(PROMPT_REASONING_STEPS_TOOLS.format(task=task, tools=tools))
            if len(completion.choices)>0:
                self.reasoning_steps = completion.choices[0].message.content

        # store messages in memory
        self.memory.system_prompt = self.get_system_prompt()
        if chat_history:
            self.memory.add_messages(chat_history)
        self.memory.add_message(self.get_user_message(task=task))

        # init agent response
        agent_response = AgentResponse()

        # global loop
        loop = 1
        while loop < MAX_TURNS:

            messages_for_model = self.memory.get()

            if stream:
                message = {
                    "content": "",
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

                content = ""
                tool_calls = []

                for chunk in self.invoke(messages=messages_for_model, stream=True):

                    chunk = ChatCompletionChunk(**chunk.model_dump())

                    agent_response.delta = None
                    agent_response.thinking = None

                    if len(chunk.choices) > 0:

                        delta = json.loads(chunk.choices[0].delta.json())

                        if delta["content"]:
                            agent_response.delta = delta["content"]
                            content += delta["content"]

                        elif delta["tool_calls"]:

                            if delta["tool_calls"][0]["id"]:
                                tool_call = delta["tool_calls"][0]
                                tool_calls.append(tool_call)

                            elif delta["tool_calls"][0]["function"]["arguments"]:
                                if len(tool_calls)>0:
                                    current_tool = len(tool_calls) - 1
                                    tool_calls[current_tool]["function"]["arguments"] += delta["tool_calls"][0]["function"]["arguments"]
                                    if delta["tool_calls"][0]["function"]["arguments"].endswith("}") and self.show_tool_calls:
                                        arguments = json.loads(tool_calls[current_tool]["function"]["arguments"])
                                        arguments = ", ".join(arguments.values())
                                        agent_response.thinking = f"""**Calling** { tool_calls[current_tool]["function"]["name"].replace("Tool","") } on '{ arguments }'"""

                        delta.pop("role")
                        delta.pop("tool_calls")
                        merge_chunk(message, delta)
                        message["tool_calls"] = tool_calls

                    if agent_response.content or agent_response.delta or agent_response.thinking:
                        yield agent_response

                message = ChatCompletionMessage(**message)
                agent_response.content = content
            
            else:
                completion = self.invoke(messages=messages_for_model)
                message = ChatCompletionMessage(**completion.choices[0].message.model_dump())
                agent_response.content = completion.choices[0].message.content

            # add messages to the current message stack
            self.memory.add_message(Message(**message.model_dump()))

            if not message.tool_calls:
                logger.debug("Task done.")
                break

            # handle tool calls and switching agents
            tool_messages = self.handle_tool_calls(message.tool_calls)
            if len(tool_messages)>0:
                self.memory.add_messages(tool_messages)

            loop += 1

        if self.response_model:
            agent_response.content = json.loads(agent_response.content)

        agent_response.finish_reason = "stop"

        yield agent_response


    def run(
        self,
        task: Optional[str] = None,
        stream: Optional[bool] = False,
        chat_history: Optional[List] = None,
    ) ->  Union[AgentResponse, Iterator[AgentResponse]]:

        logger.debug("")
        logger.debug("------------------------------------------")
        logger.debug(f"Running Agent: { self.name }")
        logger.debug("------------------------------------------")
        logger.debug("")

        if stream:
            response = self._run(task=task, chat_history=chat_history, stream=True)
            return response
    
        response = self._run(task=task, chat_history=chat_history, stream=False)
        return next(response)
