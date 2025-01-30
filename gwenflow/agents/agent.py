
import uuid
import json
import inspect
from typing import List, Union, Optional, Any, Dict, Iterator
from collections import defaultdict
from pydantic import BaseModel, model_validator, field_validator, Field, UUID4
from datetime import datetime

from gwenflow.llms import ChatOpenAI
from gwenflow.types import ChatCompletionMessage, ChatCompletionMessageToolCall
from gwenflow.tools import BaseTool
from gwenflow.memory import ChatMemoryBuffer
from gwenflow.agents.types import AgentResponse
from gwenflow.agents.utils import merge_chunk
from gwenflow.agents.prompts import PROMPT_TOOLS, PROMPT_STEPS
from gwenflow.utils import logger


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
    steps: Optional[List[str]] = []
    follow_steps: bool = False
    is_react: bool = False
 
    # --- Agent Model and Tools
    llm: Optional[Any] = Field(None, validate_default=True)
    tools: List[BaseTool] = []
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    show_tool_calls: bool = False

    # --- Task, Context and Memory
    context_vars: Optional[List[str]] = []
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
    
    def get_system_message(self, context: Optional[Any] = None):
        """Return the system message for the Agent."""

        if self.system_prompt is not None:
            return dict(role="system", content=self.system_prompt)

        system_message_lines = []

        # name, role and description
        txt = f"You are an AI agent named '{self.name}'."
        if self.role:
            txt += f" Your role is: {self.role.strip('.')}."
        if self.description:
            txt += f" {self.description.strip('.')}."
        system_message_lines.append(f"{txt}\n")

        if self.add_datetime_to_instructions:
            txt = f"The current date and time is:\n<current_date>\n{ datetime.now() }\n</current_date>\n"
            system_message_lines.append(txt)

        # tools
        if self.tools:
            tools = ",".join(self.get_tool_names())
            if self.is_react:
                tools = self.get_tools_text_schema()
            system_message_lines.append(PROMPT_TOOLS.format(tools=tools).strip())
            system_message_lines.append("")

        # instructions
        instructions = []        
        if self.response_model:
            instructions.append("Use JSON to format your answers.")
        elif self.markdown:
            instructions.append("Use markdown to format your answers.")
        if self.tools is not None:
            instructions.append("Only use the tools you are provided.")
        if context is not None:
            instructions.append("Always prefer information from the provided context over your own knowledge.")

        if len(self.instructions) > 0:
            instructions += self.instructions

        if len(instructions) > 0:
            system_message_lines.append("## Guidelines:")
            system_message_lines.extend([f"- {instruction}" for instruction in instructions])
            system_message_lines.append("")

        if self.response_model:
            system_message_lines.append("## Provide your output using the following JSON schema:")
            system_message_lines.append("<json_schema>")
            system_message_lines.append(json.dumps(self.response_model, indent=4))
            system_message_lines.append("</json_schema>")
            system_message_lines.append("")

        if self.follow_steps:
            if len(self.steps) > 0:
                system_message_lines.append("## Follow these steps:\n")
                system_message_lines.extend([f"{i+1}. {step}" for i, step in enumerate(self.steps)])
                system_message_lines.append("")
            else:
                system_message_lines.append(PROMPT_STEPS.strip())
                system_message_lines.append("")

        # final system prompt
        if len(system_message_lines) > 0:
            return dict(role="system", content=("\n".join(system_message_lines)).strip())
        
        return None


    def format_context(self, context):
        prompt  = "Use the following information from the knowledge base if it helps:\n\n<context>\n"

        if isinstance(context, str):
            prompt += context + "\n"

        elif isinstance(context, dict):
            for key in context.keys():
                prompt += f"<{key}>\n"
                prompt += context.get(key) + "\n"
                prompt += f"</{key}>\n\n"

        prompt += "</context>\n\n"

        return prompt

    def get_user_message(self, task: Optional[str] = None, context: Optional[Any] = None):
        """Return the user message for the Agent."""

        if not task and not context:
            raise ValueError("You need a task or a context (or both) to run the agent!")

        prompt = ""

        if task:
            prompt += f"You have received the following task from your manager:\n<task>\n{task}\n</task>\n\n"

        if context:
            prompt += self.format_context(context)

        return { "role": "user", "content": prompt }

    
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

    def get_tool_names(self):
        return [tool.name for tool in self.tools]

    def execute_tool_call(self, tool_name: str, arguments: Dict[str, str]) -> Any:

        available_tools = self.get_tools_map()
        if tool_name not in available_tools:
            logger.error(f"Unknown tool {tool_name}, should be instead one of { available_tools.keys() }.")
            return None

        logger.debug(f"Tool call: {tool_name}({arguments})")
        observation = available_tools[tool_name].run(**arguments)

        return observation

    def handle_tool_calls(
        self,
        tool_calls: List[ChatCompletionMessageToolCall],
    ) -> List:
        
        tool_map = self.get_tools_map()

        messages = []
        for tool_call in tool_calls:

            tool_name = tool_call.function.name

            # handle missing tool case, skip to next tool
            if tool_name not in tool_map.keys():
                logger.warning(f"Unknown tool {tool_name}, should be instead one of { tool_map.keys() }.")
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "tool_name": tool_name,
                        "content": f"Observation: Error, Tool {tool_name} not found.",
                    }
                )
                continue

            arguments = json.loads(tool_call.function.arguments)
            observation = self.execute_tool_call(tool_name, arguments)
            
            if observation:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "tool_name": tool_name,
                        "content": f"Observation:\n{observation}",
                    }
                )

        return messages

    def invoke(self, messages: list, stream: bool = False) ->  Union[Any, Iterator[Any]]:

        tools = self.get_tools_openai_schema()

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
        task: Optional[str] = None,
        *,
        context: Optional[Any] = None,
        stream: Optional[bool] = False,
    ) ->  Iterator[AgentResponse]:

        messages_for_model = []

        # system messages
        system_message = self.get_system_message(context=context)
        if system_message:
            messages_for_model.append(system_message)

        # user messages
        user_message = self.get_user_message(task=task, context=context)
        if user_message:
            messages_for_model.append(user_message)
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
                            yield AgentResponse(
                                delta=delta["content"],
                                messages=None,
                                agent=self,
                                tools=self.tools,
                            )
                        elif delta["tool_calls"] and self.show_tool_calls:
                            if delta["tool_calls"][0]["function"]["name"] and not delta["tool_calls"][0]["function"]["arguments"]:
                                response = f"""**Calling:** {delta["tool_calls"][0]["function"]["name"]}"""
                                yield AgentResponse(
                                    delta=response,
                                    messages=None,
                                    agent=self,
                                    tools=self.tools,
                                )
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
                logger.debug("Task done.")
                self.memory.add_message(message_dict) # We only keep the answer in memory (not tool calls)
                break

            # handle tool calls and switching agents
            tool_response_messages = self.handle_tool_calls(message.tool_calls)
            messages_for_model.extend(tool_response_messages)

        content = messages_for_model[-1]["content"]
        if self.response_model:
            content = json.loads(content)

        yield AgentResponse(
            content=content,
            messages=messages_for_model[init_len:],
            agent=self,
            tools=self.tools,
        )


    def run(
        self,
        task: Optional[str] = None,
        *,
        context: Optional[Any] = None,
        stream: Optional[bool] = False,
        output_file: Optional[str] = None,
    ) ->  Union[AgentResponse, Iterator[AgentResponse]]:

        logger.debug("")
        logger.debug("------------------------------------------")
        logger.debug(f"Running Agent: { self.name }")
        logger.debug("------------------------------------------")
        logger.debug("")

        if stream:
            response = self._run(
                task=task,
                context=context,
                stream=True,
            )
            return response
    
        else:

            response = self._run(
                task=task,
                context=context,
                stream=False,
            )
            response = next(response)

            if output_file:
                with open(output_file, "a") as file:
                    file.write("\n")
                    file.write("---\n\n")
                    file.write(f"# Agent: { self.name }\n")
                    file.write(f"{ task }\n")
                    file.write("\n")
                    file.write(response.content)

            return response
