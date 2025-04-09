
import uuid
import json
import re
import asyncio

from typing import List, Union, Optional, Any, Dict, Iterator, Literal
from pydantic import BaseModel, model_validator, field_validator, Field, ConfigDict, UUID4

from gwenflow.logger import logger
from gwenflow.llms import ChatBase, ChatOpenAI
from gwenflow.types import Usage, Message, Document, ChatCompletionMessageToolCall
from gwenflow.tools import BaseTool
from gwenflow.memory import ChatMemoryBuffer
from gwenflow.retriever import Retriever
from gwenflow.agents.response import AgentResponse, ToolOutput
from gwenflow.agents.prompts import PROMPT_JSON_SCHEMA, PROMPT_CONTEXT, PROMPT_KNOWLEDGE
from gwenflow.tools.mcp import MCPServer, MCPUtil


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

    mcp_servers: List[MCPServer] = Field(default_factory=list)
    """A list of MCP servers that the agent can use."""

    tool_choice: Literal["auto", "required", "none"] | str | None = None
    """The tool choice to use when calling the model."""

    reasoning_model: Optional[ChatBase] = Field(None, validate_default=True)
    """Reasoning model."""

    history: ChatMemoryBuffer | None = None
    """Historcal messages for the agent."""

    tool_output: list[ToolOutput] = Field(default_factory=list)
    """A list of tool outputs."""

    max_tool_result: int = 50
    """A max result for tools."""

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
            if self.tools or self.mcp_servers:
                self.llm.tools = self.get_all_tools()
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
    
    def reason(self, input: Union[str, List[Message], List[Dict[str, str]]],) -> AgentResponse:

        if self.reasoning_model is None:
            return None
        
        logger.debug("Reasoning...")

        reasoning_agent= Agent(
            name="ReasoningAgent",
            instructions=[
                "You are a meticulous and thoughtful assistant that solves a problem by thinking through it step-by-step.",
                "Carefully analyze the task by spelling it out loud.",
                "Then break down the problem by thinking through it step by step and develop multiple strategies to solve the problem."
                "Work through your plan step-by-step, executing any tools as needed for each step.",
                "Do not call any tool or try to solve the problem yourself.",
                "Your task is to provide a plan step-by-step, not to solve the problem yourself.",
            ],
            llm=self.reasoning_model,
            tools=self.tools
        )
        
        response = reasoning_agent.run(input)

        # only keep text outside <think>
        reasoning_content = re.sub(r'<think>.*?</think>', '', response.content, flags=re.DOTALL)
        reasoning_content = reasoning_content.strip()
        if not reasoning_content:
            return None
        
        self.history.add_message(
            Message(
                role="assistant",
                content=f"I have worked through this problem in-depth and my reasoning is summarized below.\n\n{reasoning_content}"
            )
        )

        logger.debug("Thought:\n" + reasoning_content)

        return response

    def get_all_tools(self) -> list[BaseTool]:
        """All agent tools, including MCP tools and function tools."""
        tools = self.tools
        if self.mcp_servers:
            mcp_tools = asyncio.run(MCPUtil.get_all_function_tools(self.mcp_servers))
            # tools += MCPUtil.get_all_function_tools(self.mcp_servers)
            tools += mcp_tools
        return tools
    
    def _get_tool_output(self, text: str) -> list:
        try:
            _data = json.loads(text)
            if isinstance(_data, str):
                _data = [ {"content": _data} ]
            elif isinstance(_data, dict):
                _data = [ _data ]
            elif not isinstance(_data, list):
                logger.warning("Cannot read tool output.")
            return _data
        except Exception as e:
            logger.warning(e)
        return None

    def run_tool(self, tool_call) -> Message:

        if isinstance(tool_call, dict):
            tool_call = ChatCompletionMessageToolCall(**tool_call)
    
        tool_map  = {tool.name: tool for tool in self.get_all_tools()}
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
            tool_output = tool_map[tool_name].run(**function_args)

            if tool_output:

                # keep output
                tool_output.id   = tool_call.id
                tool_output.name = tool_name
                self.tool_output.append(tool_output)

                # result output max results
                return tool_output.to_message(max_result=self.max_tool_result)

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
        # results = asyncio.run(self.aexecute_tool_calls(tool_calls))        
        results = []
        for tool_call in tool_calls:
            result = self.run_tool(tool_call)
            if result:
                results.append(result.to_dict())
            
        return results

    def _get_thinking(self, tool_calls) -> str:
        thinking = []
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                tool_call = tool_call.model_dump()
            thinking.append(f"""**Calling** { tool_call["function"]["name"].replace("Tool","") } on '{ tool_call["function"]["arguments"] }'""")
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

        # documents
        last_tool_output_index = len(self.tool_output)

        # init agent response
        agent_response = AgentResponse()

        # history
        self.history.system_prompt = self.get_system_prompt(task=task, context=context)
        self.history.add_messages(messages)

        # add reasoning
        if self.reasoning_model:
            messages_for_reasoning_model = [m.to_dict() for m in self.history.get()]
            reasoning_agent_response = self.reason(messages_for_reasoning_model)
            usage = (
                Usage(
                    requests=1,
                    input_tokens=reasoning_agent_response.usage.input_tokens,
                    output_tokens=reasoning_agent_response.usage.output_tokens,
                    total_tokens=reasoning_agent_response.usage.total_tokens,
                )
                if reasoning_agent_response.usage
                else Usage()
            )
            agent_response.usage.add(usage)
    
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
            if tool_calls and self.get_all_tools():
                tool_messages = self.execute_tool_calls(tool_calls=tool_calls)
                if len(tool_messages)>0:
                    self.history.add_messages(tool_messages)
        
        # format response
        if self.response_model:
            agent_response.content = json.loads(agent_response.content)

        agent_response.tool_output = self.tool_output[last_tool_output_index:]
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

        # documents
        last_tool_output_index = len(self.tool_output)

        # init agent response
        agent_response = AgentResponse()

        # history
        self.history.system_prompt = self.get_system_prompt(task=task, context=context)
        self.history.add_messages(messages)

        # add reasoning
        if self.reasoning_model:
            messages_for_reasoning_model = [m.to_dict() for m in self.history.get()]
            reasoning_agent_response = self.reason(messages_for_reasoning_model)
            usage = (
                Usage(
                    requests=1,
                    input_tokens=reasoning_agent_response.usage.input_tokens,
                    output_tokens=reasoning_agent_response.usage.output_tokens,
                    total_tokens=reasoning_agent_response.usage.total_tokens,
                )
                if reasoning_agent_response.usage
                else Usage()
            )
            agent_response.usage.add(usage)

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
            if tool_calls and self.get_all_tools():
                tool_messages = self.execute_tool_calls(tool_calls=tool_calls)
                if len(tool_messages)>0:
                    self.history.add_messages(tool_messages)
        
        # format response
        if self.response_model:
            agent_response.content = json.loads(agent_response.content)

        agent_response.tool_output = self.tool_output[last_tool_output_index:]
        agent_response.finish_reason = "stop"

        yield agent_response

    async def arun(
        self,
        input: Union[str, List[Message], List[Dict[str, str]]],
        context: Optional[Union[str, Dict[str, str]]] = None,
    ) -> AgentResponse:
        # loop = asyncio.new_event_loop()
        # return loop.run_until_complete(self.run(input=input, context=context))
        return self.run(input=input, context=context)
