import asyncio
import json
import re
import uuid
from typing import Any, AsyncIterator, Dict, Iterator, List, Literal, Optional, Union

from openai.types.chat import ChatCompletionMessageToolCall
from pydantic import (
    UUID4,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from gwenflow.agents.prompts import PROMPT_CONTEXT, PROMPT_JSON_SCHEMA, PROMPT_KNOWLEDGE
from gwenflow.llms import ChatBase, ChatOpenAI
from gwenflow.logger import logger
from gwenflow.memory import ChatMemoryBuffer
from gwenflow.retriever import Retriever
from gwenflow.telemetry import Tracer
from gwenflow.tools import BaseTool
from gwenflow.tools.mcp import MCPServer, MCPUtil
from gwenflow.types import (
    AgentResponse,
    ItemHelpers,
    Message,
    ResponseOutputItem,
    ToolCall,
    Usage,
    UsageDetails,
)

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

    reasoning_model: Optional[bool] = Field(None, validate_default=True)
    """Reasoning model."""

    history: ChatMemoryBuffer | None = None
    """Historcal messages for the agent."""

    retriever: Optional[Retriever] = None
    """Retriever for the agent."""

    team: List["Agent"] | None = None
    """Team of agents."""

    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    """Session ID for the agent."""

    parent_flow_id: Optional[str] = None
    """Parent flow ID for topology tracking."""

    depends_on: List[str] = Field(default_factory=list)
    """Dependencies on other agents."""

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

    def _convert_response_to_message(self, response: Any) -> Message:
        text_parts = []
        tool_calls = []
        reasoning_parts = []

        for item in response.output:
            if item.type == "message":
                for part in item.content:
                    if hasattr(part, "type"):
                        if part.type == "text":
                            text_parts.append(part.text)
                    elif isinstance(part, str):
                        text_parts.append(part)
                    elif isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))

            elif item.type == "function_call":
                tool_calls.append({
                    "id": item.call_id,
                    "type": "function",
                    "function": {
                        "name": item.name,
                        "arguments": item.arguments
                    }
                })

            elif item.type == "reasoning":
                content = (
                    getattr(item, "text", None) or
                    getattr(item, "summary", None) or
                    ""
                )
                if content and content != "none":
                    reasoning_parts.append(content)

        return Message(
            role="assistant",
            content="".join(text_parts) if text_parts else None,
            tool_calls=tool_calls if tool_calls else None,
            thinking="".join(reasoning_parts) if reasoning_parts else None
        )

    def _normalize_to_message(self, response: Any) -> Message:
        if hasattr(response, "choices"):
            raw_msg = response.choices[0].message
            return Message(
                role=raw_msg.role,
                content=raw_msg.content,
                tool_calls=raw_msg.tool_calls,
            )

        elif hasattr(response, "output"):
            return self._convert_response_to_message(response)

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

    @Tracer.agent(name="ReasoningStep")
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

        from opentelemetry import trace
        current_span = trace.get_current_span()
        current_id = current_span.get_span_context().span_id

        reasoning_agent.parent_flow_id = hex(current_id)
        reasoning_agent.depends_on = [self.name]
        response = reasoning_agent.run(input)

        # only keep text outside <think>
        reasoning_content = re.sub(r"<think>.*?</think>", "", response.content, flags=re.DOTALL)
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

    @Tracer.agent(name="ReasoningStep")
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

        from opentelemetry import trace
        current_id = trace.get_current_span().get_span_context().span_id
        reasoning_agent.parent_flow_id = hex(current_id)
        reasoning_agent.depends_on = [self.name]

        response = await reasoning_agent.arun(input)

        # only keep text outside <think>
        reasoning_content = re.sub(r"<think>.*?</think>", "", response.content, flags=re.DOTALL)
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

    @Tracer.tool()
    def run_tool(self, tool_call: ToolCall) -> Message:
        tool_map = {tool.name: tool for tool in self.get_all_tools()}

        tool_name = tool_call.function
        if hasattr(tool_name, 'name'):
            tool_name = tool_name.name

        if tool_name not in tool_map:
            logger.error(f"Tool {tool_name} does not exist")
            return Message(
                role="tool",
                tool_call_id=tool_call.id,
                tool_name=tool_name,
                content=f"Tool {tool_name} does not exist",
            )

        try:
            logger.debug(f"Tool call: {tool_name}({tool_call.arguments})")
            tool = tool_map[tool_name]
            tool_response = tool.run(**tool_call.arguments)
            if tool_response:
                return Message(
                    role="tool",
                    tool_call_id=tool_call.id,
                    tool_name=tool_name,
                    content=str(tool_response),
                )

        except Exception as e:
            logger.error(f"Error executing tool '{tool_name}': {e}")

        return Message(
            role="tool",
            tool_call_id=tool_call.id,
            tool_name=tool_name,
            content=f"Error executing tool '{tool_name}'",
        )

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

        final_results_as_dicts = []
        for res in results:
            if res:
                final_results_as_dicts.append(res.to_dict())

        return final_results_as_dicts

    def _convert_openai_tool_calls(self, message: Message) -> list[ToolCall]:
        if not message.tool_calls:
            return []
        tool_calls = []
        for tc in message.tool_calls:
            if isinstance(tc, dict):
                tool_calls.append(
                    ToolCall(
                        id=tc.get("id"),
                        function=tc["function"]["name"],
                        arguments=json.loads(tc["function"]["arguments"]) if isinstance(tc["function"]["arguments"], str) else tc["function"]["arguments"]
                    )
                )
            else:
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        function=tc.function.name,
                        arguments=json.loads(tc.function.arguments)
                    )
                )
        return tool_calls

    def _get_thinking(self, message: Message) -> str:
        if message.thinking:
            return message.thinking

        if message.tool_calls:
            thinking = []
            for tc in message.tool_calls:
                name = tc["function"]["name"] if isinstance(tc, dict) else tc.function.name
                thinking.append(f"**Calling** {name.replace('Tool', '')}...")
            return "\n".join(thinking)

        return ""

    @Tracer.agent()
    def run(
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

        num_turns_available = DEFAULT_MAX_TURNS

        while num_turns_available > 0:
            if num_turns_available <= (DEFAULT_MAX_TURNS - 1):
                self.tool_choice = 'auto'

            num_turns_available -= 1

            messages_for_model = [m.to_dict() for m in self.history.get()]

            response = self.llm.invoke(input=messages_for_model)

            usage = (
                Usage(
                    requests=1,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    total_tokens=response.usage.total_tokens,
                    input_tokens_details=UsageDetails(
                        cached_tokens=getattr(response.usage.input_tokens_details, "cached_tokens", 0),
                        reasoning_tokens=getattr(response.usage.input_tokens_details, "reasoning_tokens", 0)
                    ),
                    output_tokens_details=UsageDetails(
                        cached_tokens=getattr(response.usage.output_tokens_details, "cached_tokens", 0),
                        reasoning_tokens=getattr(response.usage.output_tokens_details, "reasoning_tokens", 0)
                    )
                )
                if response.usage
                else Usage()
            )
            agent_response.usage.add(usage)

            if hasattr(response, "choices"):
                message = Message(**response.choices[0].message.model_dump())
            else:
                message = self._convert_response_to_message(response)

            self.history.add_message(message)

            current_thinking = self._get_thinking(message)
            if current_thinking:
                if agent_response.thinking:
                    agent_response.thinking += f"\n\n{current_thinking}"
                else:
                    agent_response.thinking = current_thinking

            if not message.tool_calls:
                agent_response.content = message.content
                agent_response.messages.append(message)
                break

            tool_calls = self._convert_openai_tool_calls(message)
            if tool_calls and self.get_all_tools():
                tool_messages = self.execute_tool_calls(tool_calls=tool_calls)
                for m in tool_messages:
                    msg_obj = Message(**m) if isinstance(m, dict) else m
                    self.history.add_message(msg_obj)
                    agent_response.messages.append(msg_obj)

        if self.response_model and agent_response.content:
            try:
                agent_response.content = json.loads(agent_response.content)
            except Exception:
                pass

        agent_response.finish_reason = "stop"
        return agent_response

    @Tracer.agent()
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
            messages_for_model = [m.to_dict() for m in self.history.get()]

            response = await self.llm.ainvoke(input=messages_for_model)

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

            self.history.add_message(response.choices[0].message.model_dump())

            if not response.choices[0].message.tool_calls:
                agent_response.content = response.choices[0].message.content
                agent_response.messages.append(Message(**response.choices[0].message.model_dump()))
                break

            agent_response.thinking = self._get_thinking(response.choices[0].message.tool_calls)

            tool_calls = response.choices[0].message.tool_calls
            if tool_calls and self.get_all_tools():
                tool_messages = await self.aexecute_tool_calls(tool_calls=tool_calls)
                for m in tool_messages:
                    self.history.add_message(m)
                    agent_response.messages.append(Message(**m))

        if self.response_model:
            agent_response.content = json.loads(agent_response.content)

        for output in agent_response.messages:
            if output.role == "tool":
                try:
                    agent_response.sources.append(
                        ResponseOutputItem(
                            id=output.tool_call_id,
                            name=output.tool_name,
                            data=json.loads(output.content),
                        )
                    )
                except Exception as e:
                    logger.warning(f"Error casting source: {e}")

        agent_response.finish_reason = "stop"

        return agent_response

    @Tracer.stream()
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

        num_turns_available = DEFAULT_MAX_TURNS

        while num_turns_available > 0:
            num_turns_available -= 1

            messages_for_model = [m.to_dict() for m in self.history.get()]

            message = Message(role="assistant", content="", delta="", tool_calls=[])
            final_tool_calls = {}

            for chunk in self.llm.stream(input=messages_for_model):
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

            message.tool_calls = [final_tool_calls[k] for k in final_tool_calls.keys()]

            self.history.add_message(message.model_dump())

            if not message.tool_calls:
                agent_response.content = message.content
                agent_response.messages.append(Message(**message.model_dump()))
                break

            agent_response.thinking = self._get_thinking(message.tool_calls)
            if agent_response.thinking:
                yield agent_response

            tool_calls = self._convert_openai_tool_calls(message.tool_calls)
            if tool_calls and self.get_all_tools():
                tool_messages = self.execute_tool_calls(tool_calls=tool_calls)
                for m in tool_messages:
                    self.history.add_message(m)
                    agent_response.messages.append(Message(**m))

        if self.response_model:
            agent_response.content = json.loads(agent_response.content)

        agent_response.finish_reason = "stop"

        yield agent_response

    @Tracer.astream()
    async def arun_stream(
        self,
        input: Union[str, List[Message], List[Dict[str, str]]],
        context: Optional[Union[str, Dict[str, str]]] = None,
    ) -> AsyncIterator[AgentResponse]:

        messages = ItemHelpers.input_to_message_list(input)
        task = messages[-1].content

        agent_response = AgentResponse()

        self.history.system_prompt = self.get_system_prompt(task=task, context=context)
        self.history.add_messages(messages)

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
            messages_for_model = [m.to_dict() for m in self.history.get()]

            message = Message(role="assistant", content="", delta="", tool_calls=[])
            final_tool_calls = {}

            completions = self.llm.astream(input=messages_for_model)
            async for chunk in completions:
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

            message.tool_calls = [final_tool_calls[k] for k in final_tool_calls.keys()]

            self.history.add_message(message.model_dump())

            if not message.tool_calls:
                agent_response.content = message.content
                agent_response.messages.append(Message(**message.model_dump()))
                break

            agent_response.thinking = self._get_thinking(message.tool_calls)
            if agent_response.thinking:
                yield agent_response

            tool_calls = self._convert_openai_tool_calls(message.tool_calls)
            if tool_calls and self.get_all_tools():
                tool_messages = await self.aexecute_tool_calls(tool_calls=tool_calls)
                for m in tool_messages:
                    self.history.add_message(m)
                    agent_response.messages.append(Message(**m))

        if self.response_model:
            agent_response.content = json.loads(agent_response.content)

        agent_response.finish_reason = "stop"

        yield agent_response
