
import uuid
import json
import re
import asyncio

from typing import List, Union, Optional, Any, Dict, Iterator, Literal
from pydantic import BaseModel, model_validator, field_validator, Field, ConfigDict, UUID4

from gwenflow.logger import logger
from gwenflow.llms import ChatBase, ChatOpenAI
from gwenflow.types import Usage, Message, ChatCompletionMessageToolCall
from gwenflow.tools import BaseTool
from gwenflow.memory import ChatMemoryBuffer
from gwenflow.retriever import Retriever
from gwenflow.agents.response import AgentResponse
from gwenflow.agents.prompts import PROMPT_STEPS, PROMPT_REASONNING, PROMPT_JSON_SCHEMA, PROMPT_CONTEXT, PROMPT_KNOWLEDGE


DEFAULT_MAX_TURNS = 10


class Agent(BaseModel):

    id: UUID4 = Field(default_factory=uuid.uuid4, frozen=True)
    """The unique id of the agent."""

    name: str
    """The name of the agent."""

    description: str | None = None
    """A description of the agent, used as a handoff, so that the manager knows what it does."""

    instructions: (str | List[str] | None) = None
    """The instructions for the agent."""

    response_model: Dict | None = None
    """Response model."""

    llm: Optional[ChatBase] = Field(None, validate_default=True)
    """The model implementation to use when invoking the LLM."""

    tools: List[BaseTool] = Field(default_factory=list)
    """A list of tools that the agent can use."""

    # mcp_servers: List[MCPServer] = Field(default_factory=list)
    """A list of MCP servers that the agent can use."""

    tool_choice: Literal["auto", "required", "none"] | str | None = None
    """The tool choice to use when calling the model."""

    reasoning_model: Optional[ChatBase] = Field(None, validate_default=True)
    """Reasoning model."""

    reasoning_steps: str | None = None
    """Reasoning steps."""

    history: ChatMemoryBuffer | None = None
    """Historcal messages for the agent."""

    retriever: Optional[Retriever] = None
    """Retriever for the agent."""

    team: List["Agent"] | None = None
    """Team of agents."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    @field_validator("id", mode="before")
    @classmethod
    def deny_user_set_id(cls, v: Optional[UUID4]) -> None:
        if v:
            raise ValueError("This field is not to be set by the user.")

    @field_validator("llm", mode="before")
    @classmethod
    def set_llm(cls, v: Optional[Any]) -> Any:
        llm = v or ChatOpenAI(model="gpt-4o-mini")
        return llm

    @model_validator(mode="after")
    def model_valid(self) -> Any:
        if self.llm:
            if self.history is None:
                token_limit = self.llm.get_context_window_size()
                self.history = ChatMemoryBuffer(token_limit=token_limit)
            if self.response_model:
                self.llm.response_format = {"type": "json_object"}
            if self.tools:
                self.llm.tools = self.tools
                self.llm.tool_choice = self.tool_choice
        return self

    def _format_context(self, context: Optional[Union[str, Dict[str, str]]]) -> str:
        text = ""
        if isinstance(context, str):
            text = f"<context>\n{ context }\n</context>\n\n"
        elif isinstance(context, dict):
            for key in context.keys():
                text += f"<{key}>\n"
                text += context.get(key) + "\n"
                text += f"</{key}>\n\n"
        return text
    
    def get_system_prompt(self, task: str, context: Optional[Union[str, Dict[str, str]]] = None,) -> str:
        """Get the system prompt for the agent."""

        prompt = "Your name is {name}.".format(name=self.name)

        # instructions
        if self.instructions:
            if isinstance(self.instructions, str):
                prompt += " {instructions}".format(instructions=self.instructions)
            elif isinstance(self.instructions, list):
                instructions = "\n".join([f"- {i}" for i in self.instructions])
                prompt+= "\n\n## Instructions:\n{instructions}".format(instructions=instructions)

        prompt += "\n\n"

        # reasoning steps
        if self.reasoning_steps:
            prompt += PROMPT_STEPS.format(reasoning_steps=self.reasoning_steps).strip()
            prompt += "\n\n"

        # response model
        if self.response_model:
            prompt += PROMPT_JSON_SCHEMA.format(json_schema=json.dumps(self.response_model, indent=4)).strip()
            prompt += "\n\n"

        # references
        if self.retriever:
            references = self.retriever.search(query=task)
            if len(references)>0:
                references = [r.content for r in references]
                prompt += PROMPT_KNOWLEDGE.format(references="\n\n".join(references)).strip()
                prompt += "\n\n"

        # context
        if context is not None:
            prompt += PROMPT_CONTEXT.format(context=self._format_context(context)).strip()
            prompt += "\n\n"

        return prompt.strip()
    
    def reason(self, task: str):

        if self.reasoning_model is None:
            return None
        
        logger.debug("Reasoning...")

        prompt = ""
        
        # if self.tools:
        #     tools = self.get_tools_text_schema()
        #     user_prompt += PROMPT_TOOLS.format(tools=tools).strip() + "\n\n"

        prompt += PROMPT_REASONNING.format(task=task)

        response = self.reasoning_model.invoke(messages=[{"role": "user", "content": prompt}])

        # only keep text outside <think>
        reasoning_content = re.sub(r'<think>.*?</think>', '', response.choices[0].message.content, flags=re.DOTALL)
        if not reasoning_content:
            return None
        
        reasoning_content = reasoning_content.strip()
        
        logger.debug("Thought:\n" + reasoning_content)

        return reasoning_content


    def run_tool(self, tool_call) -> Message:

        if isinstance(tool_call, dict):
            tool_call = ChatCompletionMessageToolCall(**tool_call)
    
        tool_map  = {tool.name: tool for tool in self.tools}
        tool_name = tool_call.function.name
                    
        if tool_name not in tool_map.keys():
            logger.error(f"Tool {tool_name} does not exist")
            return Message(
                role="tool",
                tool_call_id=tool_call.id,
                tool_name=tool_name,
                content=f"Tool {tool_name} does not exist",
            )

        try:
            function_args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse tool arguments: {e}")
            return Message(
                role="tool",
                tool_call_id=tool_call.id,
                tool_name=tool_name,
                content=f"Failed to parse tool arguments: {e}",
            )

        try:
            logger.debug(f"Tool call: {tool_name}({function_args})")
            result = tool_map[tool_name].run(**function_args)
            if result:
                return Message(
                    role="tool",
                    tool_call_id=tool_call.id,
                    tool_name=tool_name,
                    content=str(result),
                )
        except Exception as e:
            logger.error(f"Error executing tool '{tool_name}': {e}")

        return Message(
            role="tool",
            tool_call_id=tool_call.id,
            tool_name=tool_name,
            content=f"Error executing tool '{tool_name}'",
        )
    
    async def aexecute_tool_calls(self, tool_calls: List[ChatCompletionMessageToolCall]) -> List:
        tasks = []
        for tool_call in tool_calls:
            task = asyncio.create_task(asyncio.to_thread(self.run_tool, tool_call))
            tasks.append(task)

        results = await asyncio.gather(*tasks)

        return results

    def execute_tool_calls(self, tool_calls: List[ChatCompletionMessageToolCall]) -> List:        
        results = asyncio.run(self.aexecute_tool_calls(tool_calls))
        return results

    def _get_thinking(self, tool_calls) -> str:
        thinking = []
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                tool_call = tool_call.model_dump()
            arguments = json.loads(tool_call["function"]["arguments"])
            arguments = ", ".join(arguments.values())
            thinking.append(f"""**Calling** { tool_call["function"]["name"].replace("Tool","") } on '{ arguments }'""")
        if len(thinking)>0:
            return "\n".join(thinking)
        return ""
    

    def run(
        self,
        input: Union[str, List[Message], List[Dict[str, str]]],
        context: Optional[Union[str, Dict[str, str]]] = None,
    ) -> AgentResponse:

        # prepare messages and task
        messages = self.llm._cast_messages(input)
        task = messages[-1].content

        # init agent response
        agent_response = AgentResponse()

        # add reasoning steps
        if self.reasoning_model:
            self.reasoning_steps = self.reason(messages[-1].content)

        # history
        self.history.system_prompt = self.get_system_prompt(task=task, context=context)
        self.history.add_messages(messages)

        while True:

            # format messages
            messages_for_model = [m.to_dict() for m in self.history.get()]

            # call llm and tool
            response = self.llm.invoke(messages=messages_for_model)

            # usage
            usage = (
                Usage(
                    requests=1,
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens,
                )
                if response.usage
                else Usage()
            )
            agent_response.usage.add(usage)

            # keep answer in memory
            self.history.add_message(response.choices[0].message.model_dump())

            # stop if not tool call
            if not response.choices[0].message.tool_calls:
                agent_response.content = response.choices[0].message.content
                break
            
            # thinking
            agent_response.thinking = self._get_thinking(response.choices[0].message.tool_calls)

            # handle tool calls
            tool_calls = response.choices[0].message.tool_calls
            if tool_calls and self.tools:
                tool_messages = self.execute_tool_calls(tool_calls=tool_calls)
                if len(tool_messages)>0:
                    self.history.add_messages(tool_messages)
        
        # format response
        if self.response_model:
            agent_response.content = json.loads(agent_response.content)

        agent_response.finish_reason = "stop"

        return agent_response

    def run_stream(
        self,
        input: Union[str, List[Message], List[Dict[str, str]]],
        context: Optional[Union[str, Dict[str, str]]] = None,
    ) -> Iterator[AgentResponse]:

        # prepare messages and task
        messages = self.llm._cast_messages(input)
        task = messages[-1].content

        # init agent response
        agent_response = AgentResponse()

        # add reasoning steps
        if self.reasoning_model:
            self.reasoning_steps = self.reason(messages[-1].content)

        # history
        self.history.system_prompt = self.get_system_prompt(task=task, context=context)
        self.history.add_messages(messages)

        while True:

            # format messages
            messages_for_model = [m.to_dict() for m in self.history.get()]

            # call llm and tool
            message = Message(role="assistant", content="", delta="", tool_calls=[])

            for chunk in self.llm.stream(messages=messages_for_model):

                # usage
                usage = (
                    Usage(
                        requests=1,
                        input_tokens=chunk.usage.prompt_tokens,
                        output_tokens=chunk.usage.completion_tokens,
                        total_tokens=chunk.usage.total_tokens,
                    )
                    if chunk.usage
                    else Usage()
                )
                agent_response.usage.add(usage)

                if not chunk.choices or not chunk.choices[0].delta:
                    continue

                delta = chunk.choices[0].delta

                agent_response.content = None
                agent_response.thinking = None

                if delta.content:
                    agent_response.content = delta.content

                if delta.tool_calls:
                    if delta.tool_calls[0].id:
                        message.tool_calls.append(delta.tool_calls[0].model_dump())
                    if delta.tool_calls[0].function.arguments:
                        current_tool = len(message.tool_calls) - 1
                        message.tool_calls[current_tool]["function"]["arguments"] += delta.tool_calls[0].function.arguments

                yield agent_response

            # keep answer in memory
            self.history.add_message(message.model_dump())

            # stop if not tool call
            if not message.tool_calls:
                agent_response.content = message.content
                break

            # thinking
            agent_response.thinking = self._get_thinking(message.tool_calls)
            if agent_response.thinking:
                yield agent_response

            # handle tool calls
            tool_calls = message.tool_calls
            if tool_calls and self.tools:
                tool_messages = self.execute_tool_calls(tool_calls=tool_calls)
                if len(tool_messages)>0:
                    self.history.add_messages(tool_messages)
        
        # format response
        if self.response_model:
            agent_response.content = json.loads(agent_response.content)

        agent_response.finish_reason = "stop"

        return agent_response
