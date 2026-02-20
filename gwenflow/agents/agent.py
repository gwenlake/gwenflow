import asyncio
import json
import re
import uuid
from typing import Any, AsyncIterator, Dict, Iterator, List, Literal, Optional, Union

from typing import List, Union, Optional, Any, Dict, Iterator, Literal, AsyncIterator
from pydantic import BaseModel, model_validator, field_validator, Field, ConfigDict, UUID4

from gwenflow.logger import logger
from gwenflow.llms import ChatBase, ChatOpenAI
from gwenflow.types import Usage, Message, AgentResponse, ItemHelpers, ToolCall, ToolResponse
from gwenflow.tools import BaseTool
from gwenflow.memory import ChatMemoryBuffer
from gwenflow.retriever import Retriever
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

    system_prompt: str | None = None
    """"System prompt"""

    instructions: str | List[str] | None = None
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
            else:
                self.llm.tools = None
                self.llm.tool_choice = None
        return self

    def _format_context(self, context: Optional[Union[str, Dict[str, str]]]) -> str:
        text = ""
        if isinstance(context, str):
            text = f"<context>\n{context}\n</context>\n\n"
        elif isinstance(context, dict):
            for key in context.keys():
                text += f"<{key}>\n"
                text += context.get(key) + "\n"
                text += f"</{key}>\n\n"
        return text

    def get_system_prompt(
        self,
        task: str,
        context: Optional[Union[str, Dict[str, str]]] = None,
    ) -> str:
        """Get the system prompt for the agent."""
        if self.system_prompt:
            return self.system_prompt

        prompt = "Your name is {name}.".format(name=self.name)

        if self.instructions:
            if isinstance(self.instructions, str):
                prompt += " {instructions}".format(instructions=self.instructions)
            elif isinstance(self.instructions, list):
                instructions = "\n".join([f"- {i}" for i in self.instructions])
                prompt += "\n\n## Instructions:\n{instructions}".format(instructions=instructions)

        prompt += "\n\n"

        if self.response_model:
            prompt += PROMPT_JSON_SCHEMA.format(json_schema=json.dumps(self.response_model, indent=4)).strip()
            prompt += "\n\n"

        if self.retriever:
            references = self.retriever.search(query=task)
            if len(references) > 0:
                references = [r.content for r in references]
                prompt += PROMPT_KNOWLEDGE.format(references="\n\n".join(references)).strip()
                prompt += "\n\n"

        if context is not None:
            prompt += PROMPT_CONTEXT.format(context=self._format_context(context)).strip()
            prompt += "\n\n"

        return prompt.strip()

    def reason(
        self,
        input: Union[str, List[Message], List[Dict[str, str]]],
    ) -> AgentResponse:
        if self.reasoning_model is None:
            return None

        logger.debug("Reasoning...")

        reasoning_agent = Agent(
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
            tools=self.tools,
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
                content=f"I have worked through this problem in-depth and my reasoning is summarized below.\n\n{reasoning_content}",
            )
        )

        logger.debug("Thought:\n" + reasoning_content)

        return response

    async def areason(
        self,
        input: Union[str, List[Message], List[Dict[str, str]]],
    ) -> AgentResponse:
        if self.reasoning_model is None:
            return None

        logger.debug("Reasoning...")

        reasoning_agent = Agent(
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
            tools=self.tools,
        )

        response = await reasoning_agent.run(input)

        # only keep text outside <think>
        reasoning_content = re.sub(r'<think>.*?</think>', '', response.content, flags=re.DOTALL)
        reasoning_content = reasoning_content.strip()
        if not reasoning_content:
            return None

        self.history.add_message(
            Message(
                role="assistant",
                content=f"I have worked through this problem in-depth and my reasoning is summarized below.\n\n{reasoning_content}",
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
    
    def run_tool(self, tool_call: ToolCall) -> Message:
    
        tool_execution = ToolResponse(
            tool_call_id=tool_call.id,
            tool_name=tool_call.function.name,
        )

        tool_map  = {tool.name: tool for tool in self.get_all_tools()}

        if tool_call.function.name not in tool_map.keys():
            logger.error(f"Tool {tool_call.function} does not exist")
            tool_execution.result = f"Tool {tool_call.function} does not exist"
            return tool_execution.to_message()

        try:
            tool = tool_map[tool_call.function.name]
            arguments = json.loads(tool_call.function.arguments)
            logger.debug(f"Tool call: {tool_call.function}({arguments})")
            tool_execution.result = tool.run(**arguments)
            if tool_execution.result:
                return tool_execution.to_message()

        except Exception as e:
            logger.error(f"Error executing tool '{tool_call.function}': {e}")

        tool_execution.result = f"Error executing tool '{tool_call.function}'"
        return tool_execution.to_message()

    def execute_tool_calls(self, tool_calls: List[ToolCall]) -> List:
        # results = asyncio.run(self.aexecute_tool_calls(tool_calls))
        results = []
        for tool_call in tool_calls:
            result = self.run_tool(tool_call)
            if result:
                results.append(result.to_dict())

        return results

    async def aexecute_tool_calls(self, tool_calls: List[ToolCall]) -> List:
        tasks = []
        for tool_call in tool_calls:
            task = asyncio.create_task(asyncio.to_thread(self.run_tool, tool_call))
            tasks.append(task)

        results = await asyncio.gather(*tasks)

        return results

    def run(
            self,
            input: Union[str, List[Message], List[Dict[str, str]]],
            context: Optional[Union[str, Dict[str, str]]] = None,
    ) -> AgentResponse:

        # prepare messages and task
        messages = ItemHelpers.input_to_message_list(input)
        task = messages[-1].content

        # init agent response
        agent_response = AgentResponse()

        # history
        self.history.system_prompt = self.get_system_prompt(task=task, context=context)
        self.history.add_messages(messages)

        # add reasoning
        if self.reasoning_model:
            messages_for_reasoning_model = [m.to_dict() for m in self.history.get()]
            response = self.reason(messages_for_reasoning_model)
            agent_response.reasoning_content = response.reasoning_content
            agent_response.usage.add(response.usage)

        num_turns_available = DEFAULT_MAX_TURNS

        while num_turns_available > 0:

            num_turns_available -= 1

            # format messages
            messages_for_model = [m.to_dict() for m in self.history.get()]

            # call llm and tool
            response = self.llm.invoke(input=messages_for_model)

            # usage
            agent_response.usage.add(response.usage)

            # keep answer in memory
            tool_calls = [t.model_dump() for t in response.tool_calls]
            _message = Message(role=response.role, content=response.content, tool_calls=tool_calls)
            self.history.add_message(_message.model_dump())

            # stop if not tool call
            if not response.tool_calls:
                agent_response.content = response.content
                agent_response.messages.append(Message(**_message.model_dump()))
                break

            # handle tool calls
            if response.tool_calls and self.get_all_tools():
                tool_messages = self.execute_tool_calls(tool_calls=response.tool_calls)
                for m in tool_messages:
                    self.history.add_message(m)
                    agent_response.messages.append(Message(**m))

            if self.tool_choice == "required":
                self.tool_choice = "auto"

        # format response
        if self.response_model:
            agent_response.parsed = json.loads(agent_response.content)

        agent_response.finish_reason = "stop"

        return agent_response

    async def arun(
            self,
            input: Union[str, List[Message], List[Dict[str, str]]],
            context: Optional[Union[str, Dict[str, str]]] = None,
    ) -> AgentResponse:

        messages = ItemHelpers.input_to_message_list(input)
        task = messages[-1].content

        agent_response = AgentResponse()

        self.history.system_prompt = self.get_system_prompt(task=task, context=context)
        self.history.add_messages(messages)

        if self.reasoning_model:
            messages_for_reasoning_model = [m.to_dict() for m in self.history.get()]
            reasoning_agent_response = await self.areason(messages_for_reasoning_model)
            agent_response.reasoning_content = response.reasoning_content
            agent_response.usage.add(response.usage)

        num_turns_available = DEFAULT_MAX_TURNS

        while num_turns_available > 0:

            num_turns_available -= 1

            messages_for_model = [m.to_dict() for m in self.history.get()]

            response = await self.llm.ainvoke(input=messages_for_model)

            # usage
            agent_response.usage.add(response.usage)

            tool_calls = [t.model_dump() for t in response.tool_calls]
            _message = Message(role=response.role, content=response.content, tool_calls=tool_calls)
            self.history.add_message(_message.model_dump())

            if not response.tool_calls:
                agent_response.content = response.content
                agent_response.messages.append(Message(**_message.model_dump()))
                break

            # handle tool calls
            if response.tool_calls and self.get_all_tools():
                tool_messages = await self.aexecute_tool_calls(tool_calls=response.tool_calls)
                for m in tool_messages:
                    self.history.add_message(m)
                    agent_response.messages.append(Message(**m))

            if self.tool_choice == "required":
                self.tool_choice = "auto"

        if self.response_model:
            agent_response.content = json.loads(agent_response.content)

        # format response
        if self.response_model:
            agent_response.parsed = json.loads(agent_response.content)

        agent_response.finish_reason = "stop"

        return agent_response

    def run_stream(
        self,
        input: Union[str, List[Message], List[Dict[str, str]]],
        context: Optional[Union[str, Dict[str, str]]] = None,
    ) -> Iterator[AgentResponse]:

        messages = ItemHelpers.input_to_message_list(input)
        task = messages[-1].content

        agent_response = AgentResponse()

        self.history.system_prompt = self.get_system_prompt(task=task, context=context)
        self.history.add_messages(messages)

        if self.reasoning_model:
            messages_for_reasoning_model = [m.to_dict() for m in self.history.get()]
            reasoning_response = self.reason(messages_for_reasoning_model)
            agent_response.reasoning_content = reasoning_response.reasoning_content
            agent_response.usage.add(reasoning_response.usage)

        num_turns_available = DEFAULT_MAX_TURNS

        while num_turns_available > 0:
            num_turns_available -= 1

            messages_for_model = [m.to_dict() for m in self.history.get()]

            full_content = ""
            final_tool_calls: List[ToolCall] = []

            for chunk in self.llm.stream(input=messages_for_model):
                if chunk.usage:
                    agent_response.usage.add(chunk.usage)

                agent_response.content = None

                if chunk.content:
                    agent_response.content = chunk.content
                    full_content += chunk.content

                if chunk.tool_calls:
                    final_tool_calls = chunk.tool_calls

                yield agent_response

            tool_calls_dicts = [t.model_dump() for t in final_tool_calls]
            _message = Message(role="assistant", content=full_content, tool_calls=tool_calls_dicts)
            self.history.add_message(_message.model_dump())

            if not final_tool_calls:
                agent_response.content = full_content
                agent_response.messages.append(Message(**_message.model_dump()))
                break

            if final_tool_calls and self.get_all_tools():
                tool_messages = self.execute_tool_calls(tool_calls=final_tool_calls)
                for m in tool_messages:
                    self.history.add_message(m)
                    agent_response.messages.append(Message(**m))

            if self.tool_choice == "required":
                self.tool_choice = "auto"

        if self.response_model:
            try:
                agent_response.parsed = json.loads(full_content)
            except Exception as e:
                logger.error(f"Failed to parse JSON response: {e}")
                agent_response.parsed = {"error": f"Parse error: {str(e)}"}

        agent_response.finish_reason = "stop"

        yield agent_response

    async def arun_stream(
        self,
        input: Union[str, List[Message], List[Dict[str, str]]],
        context: Optional[Union[str, Dict[str, str]]] = None,
    ) -> AsyncIterator["AgentResponse"]:
        
        # prepare messages and task
        messages = ItemHelpers.input_to_message_list(input)
        task = messages[-1].content

        # init agent response
        agent_response = AgentResponse()

        # history
        self.history.system_prompt = self.get_system_prompt(task=task, context=context)
        self.history.add_messages(messages)

        # add reasoning
        if self.reasoning_model:
            messages_for_reasoning_model = [m.to_dict() for m in self.history.get()]
            reasoning_agent_response = await self.areason(messages_for_reasoning_model)
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

        num_turns_available = DEFAULT_MAX_TURNS

        while num_turns_available > 0:

            num_turns_available -= 1
            
            # format messages
            messages_for_model = [m.to_dict() for m in self.history.get()]

            # call llm and tool
            message = Message(role="assistant", content="", delta="", tool_calls=[])
            final_tool_calls = {}

            async for chunk in await self.llm.astream(input=messages_for_model):

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
                
                for tool_call in delta.tool_calls or []:
                    index = tool_call.index
                    if index not in final_tool_calls:
                        final_tool_calls[index] = tool_call.model_dump()
                    final_tool_calls[index]["function"]["arguments"] += tool_call.function.arguments 

                yield agent_response

            # convert tool_calls
            message.tool_calls = [final_tool_calls[k] for k in final_tool_calls.keys()]

            # keep answer in memory
            self.history.add_message(message.model_dump())

            # stop if not tool call
            if not message.tool_calls:
                agent_response.content = message.content
                agent_response.messages.append(Message(**message.model_dump()))
                break

            # thinking
            agent_response.thinking = self._get_thinking(message.tool_calls)
            if agent_response.thinking:
                yield agent_response

            # handle tool calls
            tool_calls = self._convert_openai_tool_calls(message.tool_calls)
            if tool_calls and self.get_all_tools():
                tool_messages = await self.aexecute_tool_calls(tool_calls=tool_calls) 
                for m in tool_messages:
                    self.history.add_message(m)
                    agent_response.messages.append(Message(**m))
            
        # format response
        if self.response_model:
            agent_response.content = json.loads(agent_response.content)

        agent_response.finish_reason = "stop"

        yield agent_response