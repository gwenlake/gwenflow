
import uuid
import json
import inspect
import asyncio

from typing import List, Union, Optional, Any, Dict, Iterator
from collections import defaultdict
from pydantic import BaseModel, model_validator, field_validator, Field, UUID4
from datetime import datetime

from gwenflow.llms import ChatBase, ChatOpenAI
from gwenflow.types import Message, ChatCompletionChunk
from gwenflow.tools import BaseTool
from gwenflow.memory import ChatMemoryBuffer
from gwenflow.agents.types import AgentResponse
from gwenflow.agents.prompts import PROMPT_TOOLS, PROMPT_STEPS, PROMPT_GUIDELINES, PROMPT_JSON_SCHEMA, PROMPT_TOOLS_REACT_GUIDELINES, PROMPT_TASK, PROMPT_REASONING_STEPS_TOOLS
from gwenflow.utils import logger
from gwenflow.types import Message


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
    llm: Optional[ChatBase] = Field(None, validate_default=True)
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
    
    def get_system_prompt(self, task: str) -> str:
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

        # Task
        # prompt += PROMPT_TASK.format(task=task)

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

        if self.context:
            prompt += self.get_context()

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

    # def get_user_message(self, task: Optional[str] = None) -> Message:
    #     """Return the user message for the Agent."""
    #     prompt = PROMPT_TASK.format(task=task)
    #     return Message(role="user", content=prompt)

    def get_tools_openai_schema(self):
        return [tool.openai_schema for tool in self.tools]

    def get_tools_text_schema(self) -> str:
        descriptions = []
        for tool in self.tools:
            sig = inspect.signature(tool._run)
            description = f"{tool.name}{sig} - {tool.description}"
            descriptions.append(description)
        return "\n".join(descriptions)

    def get_tools_map(self):
        return {tool.name: tool for tool in self.tools}

    def get_tools_names(self):
        return [tool.name for tool in self.tools]

    def handle_tool_call(self, tool_call) -> Message:
   
        if not isinstance(tool_call, dict):
            tool_call = tool_call.model_dump()
            
        tool_map  = self.get_tools_map()
        tool_name = tool_call["function"]["name"]
                    
        if tool_name not in tool_map.keys():
            logger.error(f"Tool {tool_name} does not exist")
            return Message(
                role="tool",
                tool_call_id=tool_call["id"],
                tool_name=tool_name,
                content=f"Tool {tool_name} does not exist",
            )

        try:
            function_args = json.loads(tool_call["function"]["arguments"])
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse tool arguments: {e}")
            return Message(
                role="tool",
                tool_call_id=tool_call["id"],
                tool_name=tool_name,
                content=f"Failed to parse tool arguments: {e}",
            )

        try:
            logger.debug(f"Tool call: {tool_name}({function_args})")
            observation = tool_map[tool_name].run(**function_args)
            if observation:
                return Message(
                    role="tool",
                    tool_call_id=tool_call["id"],
                    tool_name=tool_name,
                    content=f"Observation: {observation}",
                )
        except Exception as e:
            logger.error(f"Error executing tool '{tool_name}': {e}")

        return Message(
            role="tool",
            tool_call_id=tool_call["id"],
            tool_name=tool_name,
            content=f"Error executing tool '{tool_name}'",
        )


    def handle_tool_calls(self, tool_calls: list) -> List:
        
        tool_map = self.get_tools_map()
        if not tool_calls or not tool_map:
            return []
        
        messages = []
        for tool_call in tool_calls:
            observation = self.handle_tool_call(tool_call)
            if observation:
                messages.append(observation)
            
        return messages

    async def ahandle_tool_calls(self, tool_calls: list) -> List:
        
        tool_map = self.get_tools_map()
        if not tool_calls or not tool_map:
            return []

        tasks = []
        for tool_call in tool_calls:
            task = asyncio.create_task(asyncio.to_thread(self.handle_tool_call, tool_call))
            tasks.append(task)

        messages = []
        results = await asyncio.gather(*tasks)
        for observation in results:
            if observation:
                messages.append(observation)

        return messages

    def _get_thinking(self, tool_calls):
        thinking = []
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                tool_call = tool_call.model_dump()
            arguments = json.loads(tool_call["function"]["arguments"])
            arguments = ", ".join(arguments.values())
            thinking.append(f"""**Calling** { tool_call["function"]["name"].replace("Tool","") } on '{ arguments }'""")
        if len(thinking)>0:
            return "\n".join(thinking)
        return None
    
    def invoke(self, messages: list, stream: bool = False) ->  Union[Any, Iterator[Any]]:

        self.llm.tools = self.tools
        self.llm.tool_choice = self.tool_choice

        self.llm.response_format = None
        if self.response_model:
            self.llm.response_format = {"type": "json_object"}

        if stream:
            return self.llm.stream(messages=messages)
        
        return self.llm.invoke(messages=messages)

    async def ainvoke(self, messages: list, stream: bool = False) ->  Union[Any, Iterator[Any]]:

        self.llm.tools = self.tools
        self.llm.tool_choice = self.tool_choice

        self.llm.response_format = None
        if self.response_model:
            self.llm.response_format = {"type": "json_object"}

        if stream:
            return self.llm.astream(messages=messages)
        
        return self.llm.ainvoke(messages=messages)

    def _run(
        self,
        task: str,
        stream: Optional[bool] = False,
        chat_history: Optional[list] = [],
    ) ->  Iterator[AgentResponse]:

        # add reasoning steps
        if self.reasoning_model:
            tools = self.get_tools_text_schema()
            completion = self.invoke(PROMPT_REASONING_STEPS_TOOLS.format(task=task, tools=tools))
            if len(completion.choices)>0:
                self.reasoning_steps = completion.choices[0].message.content

        # prepare memory (system+history+task)
        self.memory.system_prompt = self.get_system_prompt(task=task)
        if len(chat_history)>0:
            self.memory.add_messages(chat_history)
        self.memory.add_message(Message(role="user", content=task))

        # init agent response
        agent_response = AgentResponse()

        # global loop
        while True:

            messages_for_model = self.memory.get()

            if stream:

                message = Message(role="assistant", content="", delta="", tool_calls=[])
                for chunk in self.invoke(messages=messages_for_model, stream=True):

                    chunk = ChatCompletionChunk(**chunk.model_dump())

                    agent_response.delta = None
                    agent_response.thinking = None

                    if len(chunk.choices) > 0:
                        if chunk.choices[0].delta.content:
                            agent_response.delta = chunk.choices[0].delta.content
                            message.content     += chunk.choices[0].delta.content
                            yield agent_response
                        elif chunk.choices[0].delta.tool_calls:
                            if chunk.choices[0].delta.tool_calls[0].id:
                                message.tool_calls.append(chunk.choices[0].delta.tool_calls[0].model_dump())
                            if chunk.choices[0].delta.tool_calls[0].function.arguments:
                                current_tool = len(message.tool_calls) - 1
                                message.tool_calls[current_tool]["function"]["arguments"] += chunk.choices[0].delta.tool_calls[0].function.arguments
            
            else:
                completion = self.invoke(messages=messages_for_model)
                message = Message(**completion.choices[0].message.model_dump())
                message.tool_calls = completion.choices[0].message.model_dump()["tool_calls"]

            # add messages to the current message stack
            self.memory.add_message(message)

            if not message.tool_calls:
                logger.debug("Task done.")
                agent_response.content = message.content
                break

            # thinking
            try:
                agent_response.thinking = self._get_thinking(message.tool_calls)
            except:
                pass
            if stream:
                if agent_response.thinking:
                    yield agent_response

            # handle tool calls and switching agents
            tool_messages = self.handle_tool_calls(message.tool_calls)
            if len(tool_messages)>0:
                self.memory.add_messages(tool_messages)

        if self.response_model:
            agent_response.content = json.loads(agent_response.content)

        agent_response.finish_reason = "stop"

        yield agent_response


    def run(
        self,
        task: str,
        stream: Optional[bool] = False,
        chat_history: Optional[list] = [],
    ) ->  Union[AgentResponse, Iterator[AgentResponse]]:

        logger.debug("")
        logger.debug("------------------------------------------")
        logger.debug(f"Running Agent: { self.name }")
        logger.debug("------------------------------------------")
        logger.debug("")

        if stream:
            response = self._run(task=task, stream=True, chat_history=chat_history)
            return response
    
        response = self._run(task=task, stream=False, chat_history=chat_history)
        return next(response)
