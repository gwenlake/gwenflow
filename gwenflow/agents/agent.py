import asyncio
import json
import re
import uuid
from typing import Any, AsyncIterator, Dict, Iterator, List, Literal, Optional, Union

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
    ToolCall,
    Usage,
    UsageInputDetails,
    UsageReasoning,
)
from gwenflow.types.responses.response_event import (
    ResponseContentDeltaEvent,
    ResponseContentEvent,
    ResponseEvent,
    ResponseReasoningDeltaEvent,
    ResponseReasoningEvent,
    ResponseToolCallEvent,
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

    thinking_model: Optional[bool] = Field(None, validate_default=True)
    """Thinking model."""

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
                tool_calls.append(
                    {
                        "id": getattr(item, "id", None) or getattr(item, "call_id", None),
                        "type": "function",
                        "function": {
                            "name": getattr(item, "name", None),
                            "arguments": getattr(item, "arguments", "{}"),
                        },
                    }
                )

            elif item.type == "reasoning":
                content = getattr(item, "text", None) or getattr(item, "summary", None) or ""
                if content and content != "none":
                    reasoning_parts.append(content)

        return Message(
            role="assistant",
            content="".join(text_parts) if text_parts else None,
            tool_calls=tool_calls if tool_calls else None,
            reasoning="".join(reasoning_parts) if reasoning_parts else None,
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

    def reason(
        self,
        input: Union[str, List[Message], List[Dict[str, str]]],
    ) -> AgentResponse:
        if self.thinking_model is None:
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
            llm=self.thinking_model,
            tools=self.tools,
        )

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

    async def areason(
        self,
        input: Union[str, List[Message], List[Dict[str, str]]],
    ) -> AgentResponse:
        if self.thinking_model is None:
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
            llm=self.thinking_model,
            tools=self.tools,
        )

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
            tools += mcp_tools
        return tools

    @Tracer.tool(name="Tool")
    def run_tool(self, tool_call: ToolCall) -> Message:
        tool_map = {tool.name: tool for tool in self.get_all_tools()}
        tool_name = tool_call.function

        if tool_name not in tool_map:
            return Message(role="tool", tool_call_id=tool_call.id, content=f"Error: Tool {tool_name} not found.")

        try:
            tool = tool_map[tool_name]
            result = tool.run(**tool_call.arguments)
            return Message(
                role="tool",
                tool_call_id=tool_call.id,
                tool_name=tool_name,
                content=str(result) if result else "No results found.",
            )
        except Exception as e:
            return Message(role="tool", tool_call_id=tool_call.id, content=f"Error during tool execution: {str(e)}")

    def execute_tool_calls(self, tool_calls: List[ToolCall]) -> List:
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
                        arguments=json.loads(tc["function"]["arguments"])
                        if isinstance(tc["function"]["arguments"], str)
                        else tc["function"]["arguments"],
                    )
                )
            else:
                tool_calls.append(
                    ToolCall(id=tc.id, function=tc.function.name, arguments=json.loads(tc.function.arguments))
                )
        return tool_calls

    def _get_thinking(self, message: Message) -> str:
        if message.thinking:
            return message.thinking

        if message.tool_calls and self.thinking_model:
            thinking = []
            for tc in message.tool_calls:
                name = tc["function"]["name"] if isinstance(tc, dict) else tc.function.name
                thinking.append(f"**Calling** {name.replace('Tool', '')}...")
            return "\n".join(thinking)

        return ""

    def _tool_executor_callback(self, name: str, arguments: Dict[str, Any] | str) -> str:
        if isinstance(arguments, str):
            try:
                args = json.loads(arguments)
            except json.JSONDecodeError:
                return f"Error: Invalid JSON arguments for tool {name}"
        else:
            args = arguments

        temp_tool_call = ToolCall(id="temp", function=name, arguments=args)
        result_message = self.run_tool(temp_tool_call)

        return str(result_message.content)

    @Tracer.agent(name="Agent Run")
    def run(
        self,
        input: Union[str, List[Message], List[Dict[str, str]]],
        context: Optional[Union[str, Dict[str, str]]] = None,
    ) -> AgentResponse:
        messages = ItemHelpers.input_to_message_list(input)
        task = messages[-1].content
        agent_response = AgentResponse()

        sys_prompt = self.get_system_prompt(task=task, context=context)
        self.history.system_prompt = sys_prompt
        self.history.add_messages(messages)

        if self.thinking_model:
            messages_for_thinking_model = list(self.history.get())
            reasoning_agent_response = self.reason(messages_for_thinking_model)
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
                self.tool_choice = "auto"
            num_turns_available -= 1

            messages_for_model = list(self.history.get())

            response = self.llm.invoke(input=messages_for_model, instructions=self.history.system_prompt)

            if response.usage:
                input_details = getattr(response.usage, "input_tokens_details", None)
                output_details = getattr(response.usage, "output_tokens_details", None)

                usage = Usage(
                    requests=1,
                    input_tokens=response.usage.input_tokens,
                    input_tokens_details=UsageInputDetails(cached_tokens=getattr(input_details, "cached_tokens", 0))
                    if input_details
                    else None,
                    output_tokens=response.usage.output_tokens,
                    output_tokens_details=UsageReasoning(
                        reasoning_tokens=getattr(output_details, "reasoning_tokens", 0)
                    )
                    if output_details
                    else None,
                    total_tokens=response.usage.total_tokens,
                )
                agent_response.usage.add(usage)

            new_messages_to_add: List[Message] = []

            if hasattr(response, "output"):
                new_messages_to_add = Message.extract_event_from_response_output(
                    response, tool_executor=self._tool_executor_callback
                )

            else:
                message = self._normalize_to_message(response)
                if message.content is None:
                    message.content = ""
                new_messages_to_add.append(message)

                if message.tool_calls:
                    tool_calls = self._convert_openai_tool_calls(message)
                    tool_results_dicts = self.execute_tool_calls(tool_calls=tool_calls)

                    for res_dict in tool_results_dicts:
                        if res_dict.get("content") is None:
                            res_dict["content"] = "No results found."
                        new_messages_to_add.append(Message(**res_dict))

            should_break = False

            for msg in new_messages_to_add:
                if msg.reasoning:
                    agent_response.reasoning = (
                        (agent_response.reasoning + "\n\n" + msg.reasoning) if agent_response.reasoning else None
                    )
                self.history.add_message(msg)
                agent_response.messages.append(msg)

                current_thinking = self._get_thinking(msg)
                if current_thinking:
                    agent_response.thinking = (
                        (agent_response.thinking + "\n\n" + current_thinking)
                        if agent_response.thinking
                        else current_thinking
                    )

                if msg.role == "assistant" and msg.content and not msg.tool_calls:
                    agent_response.content = msg.content
                    should_break = True
            if should_break:
                break

        agent_response.finish_reason = "stop"
        return agent_response

    @Tracer.agent(name="Agent Async Run")
    async def arun(
        self,
        input: Union[str, List[Message], List[Dict[str, str]]],
        context: Optional[Union[str, Dict[str, str]]] = None,
    ) -> AgentResponse:
        messages = ItemHelpers.input_to_message_list(input)
        task = messages[-1].content
        agent_response = AgentResponse()

        sys_prompt = self.get_system_prompt(task=task, context=context)
        self.history.system_prompt = sys_prompt
        self.history.add_messages(messages)

        if self.thinking_model:
            messages_for_thinking_model = list(self.history.get())
            reasoning_agent_response = await self.areason(messages_for_thinking_model)
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
                self.tool_choice = "auto"
            num_turns_available -= 1

            messages_for_model = list(self.history.get())

            response = await self.llm.ainvoke(input=messages_for_model, instructions=self.history.system_prompt)

            if response.usage:
                input_details = getattr(response.usage, "input_tokens_details", None)
                output_details = getattr(response.usage, "output_tokens_details", None)

                usage = Usage(
                    requests=1,
                    input_tokens=response.usage.input_tokens,
                    input_tokens_details=UsageInputDetails(cached_tokens=getattr(input_details, "cached_tokens", 0))
                    if input_details
                    else None,
                    output_tokens=response.usage.output_tokens,
                    output_tokens_details=UsageReasoning(
                        reasoning_tokens=getattr(output_details, "reasoning_tokens", 0)
                    )
                    if output_details
                    else None,
                    total_tokens=response.usage.total_tokens,
                )
                agent_response.usage.add(usage)

            new_messages_to_add: List[Message] = []

            if hasattr(response, "output"):
                new_messages_to_add = Message.extract_event_from_response_output(
                    response, tool_executor=self._tool_executor_callback
                )

            else:
                message = self._normalize_to_message(response)
                if message.content is None:
                    message.content = ""
                new_messages_to_add.append(message)

                if message.tool_calls:
                    tool_calls = self._convert_openai_tool_calls(message)
                    tool_results_dicts = await self.aexecute_tool_calls(tool_calls=tool_calls)

                    for res_dict in tool_results_dicts:
                        if res_dict.get("content") is None:
                            res_dict["content"] = "No results found."
                        new_messages_to_add.append(Message(**res_dict))

            should_break = False

            for msg in new_messages_to_add:
                if msg.reasoning:
                    agent_response.reasoning = (
                        (agent_response.reasoning + "\n\n" + msg.reasoning) if agent_response.reasoning else None
                    )
                self.history.add_message(msg)
                agent_response.messages.append(msg)

                current_thinking = self._get_thinking(msg)
                if current_thinking:
                    agent_response.thinking = (
                        (agent_response.thinking + "\n\n" + current_thinking)
                        if agent_response.thinking
                        else current_thinking
                    )

                if msg.role == "assistant" and msg.content and not msg.tool_calls:
                    agent_response.content = msg.content
                    should_break = True
            if should_break:
                break

        agent_response.finish_reason = "stop"
        return agent_response

    @Tracer.agent(name="Agent Stream")
    def run_stream(
        self,
        input: Union[str, List[Message], List[Dict[str, str]]],
        context: Optional[Union[str, Dict[str, str]]] = None,
    ) -> Iterator[AgentResponse]:
        messages = ItemHelpers.input_to_message_list(input)
        task = messages[-1].content
        agent_response = AgentResponse()

        sys_prompt = self.get_system_prompt(task=task, context=context)
        self.history.system_prompt = sys_prompt
        self.history.add_messages(messages)

        if self.thinking_model:
            messages_for_thinking_model = list(self.history.get())
            reasoning_response = self.reason(messages_for_thinking_model)
            if reasoning_response and reasoning_response.usage:
                agent_response.usage.add(reasoning_response.usage)

        num_turns_available = DEFAULT_MAX_TURNS

        while num_turns_available > 0:
            if num_turns_available <= (DEFAULT_MAX_TURNS - 1):
                self.tool_choice = "auto"
            num_turns_available -= 1

            messages_for_model = list(self.history.get())

            message = Message(role="assistant", content="", reasoning="", tool_calls=[])
            final_tool_calls = {}

            stream_gen = self.llm.stream(input=messages_for_model, instructions=self.history.system_prompt)

            for chunk in stream_gen:
                agent_response.content = None
                if hasattr(agent_response, "reasoning"):
                    agent_response.reasoning = None

                tool_call_updated = False

                # --- CAS A : LEGACY (Chunk.choices) ---
                if hasattr(chunk, "choices") and chunk.choices:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, "content") and delta.content:
                        message.content += delta.content
                        agent_response.content = delta.content

                    if hasattr(delta, "tool_calls") and delta.tool_calls:
                        tool_call_updated = True
                        for tc in delta.tool_calls:
                            idx = tc.index
                            if idx not in final_tool_calls:
                                final_tool_calls[idx] = {
                                    "id": tc.id,
                                    "type": "function",
                                    "function": {"name": tc.function.name or "", "arguments": ""},
                                }
                            if tc.function.arguments:
                                final_tool_calls[idx]["function"]["arguments"] += tc.function.arguments
                            if tc.function.name and not final_tool_calls[idx]["function"]["name"]:
                                final_tool_calls[idx]["function"]["name"] = tc.function.name

                # --- CAS B : RESPONSE API (Chunk.root) ---
                elif hasattr(chunk, "root"):
                    event = chunk.root

                    if isinstance(event, ResponseEvent):
                        if event.type in ["response.done", "response.completed"]:
                            if event.response and event.response.usage:
                                u = event.response.usage
                                agent_response.usage.add(
                                    Usage(
                                        requests=1,
                                        input_tokens=u.input_tokens,
                                        output_tokens=u.output_tokens,
                                        total_tokens=u.total_tokens,
                                        output_tokens_details=UsageReasoning(
                                            reasoning_tokens=getattr(u.output_tokens_details, "reasoning_tokens", 0)
                                        )
                                        if getattr(u, "output_tokens_details", None)
                                        else None,
                                    )
                                )
                        continue

                    elif isinstance(event, (ResponseReasoningDeltaEvent, ResponseReasoningEvent)):
                        text = getattr(event, "delta", None) or getattr(event, "text", "")
                        if text:
                            if message.reasoning is None:
                                message.reasoning = ""
                            message.reasoning += text
                            agent_response.reasoning = text

                    elif isinstance(event, (ResponseContentDeltaEvent, ResponseContentEvent)):
                        text = getattr(event, "delta", None) or getattr(event, "text", "")
                        if text:
                            message.content += text
                            agent_response.content = text

                    elif isinstance(event, ResponseToolCallEvent):
                        tool_call_updated = True
                        cid = getattr(event, "item_id", None) or getattr(event, "call_id", None)

                        if cid:
                            if cid not in final_tool_calls:
                                final_tool_calls[cid] = {
                                    "id": cid,
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""},
                                }

                            name_part = getattr(event, "name", "")
                            if name_part:
                                final_tool_calls[cid]["function"]["name"] = name_part

                            delta = getattr(event, "delta", None)

                            if delta:
                                final_tool_calls[cid]["function"]["arguments"] += delta
                            else:
                                full_args = getattr(event, "arguments", None)
                                if full_args:
                                    final_tool_calls[cid]["function"]["arguments"] = full_args

                should_yield = (
                    (agent_response.content is not None)
                    or (agent_response.reasoning is not None)
                    or (tool_call_updated)
                )

                if should_yield:
                    yield agent_response

            if final_tool_calls:
                for tc_data in final_tool_calls.values():
                    args = tc_data["function"]["arguments"]
                    if not args or not args.strip():
                        tc_data["function"]["arguments"] = "{}"

                message.tool_calls = list(final_tool_calls.values())
            else:
                message.tool_calls = None

            self.history.add_message(message)

            if not message.tool_calls:
                agent_response.content = message.content
                agent_response.messages.append(message)
                break

            tool_calls = self._convert_openai_tool_calls(message)
            tool_results = self.execute_tool_calls(tool_calls=tool_calls)

            for res_dict in tool_results:
                if res_dict.get("content") is None:
                    res_dict["content"] = "No results found."
                msg_obj = Message(**res_dict)
                self.history.add_message(msg_obj)
                agent_response.messages.append(msg_obj)

        agent_response.finish_reason = "stop"
        yield agent_response

    @Tracer.agent(name="Agent Async Stream")
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

        if self.thinking_model:
            messages_for_thinking_model = list(self.history.get())
            reasoning_agent_response = await self.areason(messages_for_thinking_model)
            if reasoning_agent_response and reasoning_agent_response.usage:
                agent_response.usage.add(reasoning_agent_response.usage)

        num_turns_available = DEFAULT_MAX_TURNS
        while num_turns_available > 0:
            if num_turns_available <= (DEFAULT_MAX_TURNS - 1):
                self.tool_choice = "auto"
            num_turns_available -= 1
            messages_for_model = list(self.history.get())

            message = Message(role="assistant", content="", reasoning="", tool_calls=[])
            final_tool_calls = {}

            async for chunk in self.llm.astream(input=messages_for_model):
                agent_response.content = None
                if hasattr(agent_response, "reasoning"):
                    agent_response.reasoning = None

                tool_call_updated = False

                if hasattr(chunk, "choices") and chunk.choices:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, "content") and delta.content:
                        message.content += delta.content
                        agent_response.content = delta.content

                    if hasattr(delta, "tool_calls") and delta.tool_calls:
                        tool_call_updated = True
                        for tc in delta.tool_calls:
                            idx = tc.index
                            if idx not in final_tool_calls:
                                final_tool_calls[idx] = tc.model_dump()
                            else:
                                final_tool_calls[idx]["function"]["arguments"] += tc.function.arguments or ""

                elif hasattr(chunk, "root"):
                    event = chunk.root

                    if isinstance(event, ResponseEvent):
                        if event.type in ["response.done", "response.completed"]:
                            if event.response and event.response.usage:
                                u = event.response.usage
                                agent_response.usage.add(
                                    Usage(
                                        requests=1,
                                        input_tokens=u.input_tokens,
                                        output_tokens=u.output_tokens,
                                        total_tokens=u.total_tokens,
                                        output_tokens_details=UsageReasoning(
                                            reasoning_tokens=getattr(u.output_tokens_details, "reasoning_tokens", 0)
                                        ),
                                    )
                                )
                        continue

                    elif isinstance(event, (ResponseReasoningDeltaEvent, ResponseReasoningEvent)):
                        text = getattr(event, "delta", None) or getattr(event, "text", "")
                        if text:
                            if message.reasoning is None:
                                message.reasoning = ""
                            message.reasoning += text

                            if hasattr(self.llm, "show_reasoning") and self.llm.show_reasoning:
                                if hasattr(agent_response, "reasoning"):
                                    agent_response.reasoning = text
                                else:
                                    agent_response.thinking = text

                    elif isinstance(event, (ResponseContentDeltaEvent, ResponseContentEvent)):
                        text = getattr(event, "delta", None) or getattr(event, "text", "")
                        if text:
                            message.content += text
                            agent_response.content = text

                    elif isinstance(event, ResponseToolCallEvent):
                        tool_call_updated = True
                        cid = getattr(event, "item_id", None) or getattr(event, "call_id", None)

                        if cid:
                            if cid not in final_tool_calls:
                                final_tool_calls[cid] = {
                                    "id": cid,
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""},
                                }

                            name_part = getattr(event, "name", "")
                            if name_part:
                                final_tool_calls[cid]["function"]["name"] = name_part

                            delta = getattr(event, "delta", None)
                            if delta:
                                final_tool_calls[cid]["function"]["arguments"] += delta
                            else:
                                full_args = getattr(event, "arguments", None)
                                if full_args:
                                    final_tool_calls[cid]["function"]["arguments"] = full_args

                has_reasoning = getattr(agent_response, "reasoning", None) or agent_response.thinking
                if agent_response.content or tool_call_updated or has_reasoning:
                    yield agent_response

            if final_tool_calls:
                for tc_data in final_tool_calls.values():
                    args = tc_data["function"]["arguments"]
                    if not args or not args.strip():
                        tc_data["function"]["arguments"] = "{}"
                message.tool_calls = list(final_tool_calls.values())
            else:
                message.tool_calls = None

            message.content = message.content or ""

            self.history.add_message(message)

            if not message.tool_calls:
                agent_response.content = message.content
                agent_response.messages.append(message)
                break

            tool_calls = self._convert_openai_tool_calls(message)
            tool_results = await self.aexecute_tool_calls(tool_calls=tool_calls)

            for res_dict in tool_results:
                if res_dict.get("content") is None:
                    res_dict["content"] = "No results found."
                msg_obj = Message(**res_dict)
                self.history.add_message(msg_obj)
                agent_response.messages.append(msg_obj)

        agent_response.finish_reason = "stop"
        yield agent_response
