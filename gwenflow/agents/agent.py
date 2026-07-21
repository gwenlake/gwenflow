import asyncio
import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Dict, Iterator, List, Literal, Optional, Type, Union

from pydantic import BaseModel, ValidationError

from gwenflow.agents.prompts import PROMPT_CONTEXT, PROMPT_JSON_SCHEMA, PROMPT_KNOWLEDGE
from gwenflow.llms import ChatBase, ChatOpenAI
from gwenflow.logger import logger
from gwenflow.memory import ChatMemoryBuffer
from gwenflow.retriever import Retriever
from gwenflow.skills import Skill, SkillsToolset
from gwenflow.telemetry import tracer
from gwenflow.tools import BaseTool, MCPServer, Tool
from gwenflow.types import (
    AgentEventCompleted,
    AgentEventContent,
    AgentEventStarted,
    AgentEventThinking,
    AgentEventToolCompleted,
    AgentEventToolStarted,
    AgentResponse,
    Message,
    ModelResponse,
    RequestUsage,
    ToolCall,
    ToolResponse,
)
from gwenflow.utils import extract_json_str


@dataclass
class Agent:
    id: str | None = None
    """The unique id of the agent."""

    name: str | None = None
    """The name of the agent."""

    description: str | None = None
    """A description of the agent, used as a handoff, so that the manager knows what it does."""

    system_prompt: str | None = None
    """System prompt."""

    instructions: str | List[str] | None = None
    """The instructions for the agent."""

    response_model: Optional[Union[Dict[str, Any], Type[BaseModel]]] = None
    """Response model."""

    llm: Optional[ChatBase] = None
    """The model implementation to use when invoking the LLM."""

    tools: List[BaseTool] = field(default_factory=list)
    """A list of tools that the agent can use."""

    mcp_servers: List[MCPServer] = field(default_factory=list)
    """A list of MCP servers that the agent can use."""

    tool_choice: Literal["auto", "required", "none"] | str | None = None
    """The tool choice to use when calling the model."""

    reasoning_model: Optional[ChatBase] = None
    """Reasoning model."""

    history: ChatMemoryBuffer | None = None
    """Historical messages for the agent."""

    retriever: Optional[Retriever] = None
    """Retriever for the agent."""

    team: List["Agent"] | None = None
    """Team of agents."""

    max_turns: Optional[int] = 100
    """Maximum turn (tool calls, llm calls) an agent can do."""

    skills: List[Skill] = field(default_factory=list)
    """Skills that extend the agent's instructions."""

    def __post_init__(self) -> None:
        if self.id is None:
            self.id = str(uuid.uuid4())
        if self.llm is None:
            self.llm = ChatOpenAI()
        if self.history is None:
            self.history = ChatMemoryBuffer(token_limit=self.llm.get_context_size())
        if self.response_model:
            self.llm.response_format = self.response_model
        if self.skills:
            self.tools = list(self.tools) + SkillsToolset(self.skills).get_tools()
        if self.team:
            self.tools = list(self.tools) + self._build_handoff_tools()
        if self.tools or self.mcp_servers:
            self.llm.tools = self.get_all_tools()
            self.llm.tool_choice = self.tool_choice
        else:
            self.llm.tools = None
            self.llm.tool_choice = None

    def _build_handoff_tools(self) -> List[BaseTool]:
        """Expose each team member as a callable tool the orchestrator can invoke."""

        def make_handoff(member: "Agent") -> Callable[[str], str]:
            def _handoff(task: str) -> str:
                """Send a task to a teammate agent and return its final answer.

                Args:
                    task: The full task or question to delegate to the teammate.
                """
                response = member.run(task)
                return response.content or ""

            return _handoff

        tools: List[BaseTool] = []
        used_names: set[str] = set()
        for member in self.team or []:
            slug = re.sub(r"\W+", "_", (member.name or "agent").strip().lower()).strip("_") or "agent"
            tool_name = f"ask_{slug}"
            n = 2
            while tool_name in used_names:
                tool_name = f"ask_{slug}_{n}"
                n += 1
            used_names.add(tool_name)

            desc = member.description or f"Delegate a task to the '{member.name or slug}' agent."
            tools.append(Tool(make_handoff(member), name=tool_name, description=desc))
        return tools

    def tool(self, func: Callable) -> Callable:
        """Decorator that registers a function as a tool on this agent instance."""
        # self.tools = list(self.tools) + [FunctionTool.from_function(func)]
        self.tools = list(self.tools) + [Tool(func)]
        self.llm.tools = self.get_all_tools()
        self.llm.tool_choice = self.tool_choice
        return func

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

    def _validate_final_response(
        self, content_to_parse: str | Dict[str, Any] | List[Any] | None
    ) -> tuple[bool, Any, Optional[str]]:
        if not (isinstance(self.response_model, type) and issubclass(self.response_model, BaseModel)):
            return True, content_to_parse, None

        try:
            if isinstance(content_to_parse, str):
                content_to_parse = json.loads(extract_json_str(content_to_parse))

            parsed_obj = self.response_model.model_validate(content_to_parse)
            return True, parsed_obj, None
        except (ValidationError, json.JSONDecodeError, TypeError, ValueError) as e:
            error_msg = (
                f"Your final response failed validation against the required schema. "
                f"Error details:\n{str(e)}\n"
                f"Please correct the errors and return ONLY the valid JSON."
            )
            return False, None, error_msg

    def _tools_token_count(self) -> int:
        """Tokens the tool schemas cost on the wire.

        Every provider sends the same three fields per tool (name, description,
        JSON-schema parameters), so a single neutral estimate covers them all.
        These tokens never appear in the message list, so the memory buffer has
        to be told to reserve them — a large MCP server can otherwise eat the
        whole context margin without the buffer noticing.
        """
        tools = self.llm.tools or []
        tokenizer_fn = self.history.tokenizer_fn
        total = 0
        for tool in tools:
            schema = {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.parameters,
            }
            total += tokenizer_fn(json.dumps(schema, default=str))
        return total

    def _prepare_history(
        self,
        task: str,
        context: Optional[Union[str, Dict[str, str]]] = None,
    ) -> None:
        """Refresh the parts of the memory budget that are settled per run.

        The system prompt and the tool schemas sent alongside every request.
        """
        self.history.system_prompt = self.get_system_prompt(task=task, context=context)
        self.history.reserved_tokens = self._tools_token_count()

    def get_system_prompt(
        self,
        task: str,
        context: Optional[Union[str, Dict[str, str]]] = None,
    ) -> str:
        """Get the system prompt for the agent."""
        if self.system_prompt:
            return self.system_prompt.strip()

        system_prompt_parts = []

        if self.instructions:
            instructions = "## Instructions:\n\n"
            if isinstance(self.instructions, str):
                instructions += "- {instructions}".format(instructions=self.instructions)
            elif isinstance(self.instructions, list):
                instructions += "\n".join([f"- {i}" for i in self.instructions])
            system_prompt_parts.append(instructions)

        if self.skills:
            skills = "## Skills:\n\n"
            skills += SkillsToolset(self.skills).get_instructions()
            system_prompt_parts.append(skills)

        if self.response_model:
            if isinstance(self.response_model, type) and issubclass(self.response_model, BaseModel):
                schema_str = json.dumps(self.response_model.model_json_schema(), indent=4)
            else:
                schema_str = json.dumps(self.response_model, indent=4)
            prompt_schema = PROMPT_JSON_SCHEMA.format(json_schema=schema_str).strip()
            system_prompt_parts.append(prompt_schema)

        if self.retriever:
            references = self.retriever.search(query=task)
            if len(references) > 0:
                references = [r.content for r in references]
                prompt_references = PROMPT_KNOWLEDGE.format(references="\n\n".join(references)).strip()
                system_prompt_parts.append(prompt_references)

        if context is not None:
            prompt_context = PROMPT_CONTEXT.format(context=self._format_context(context)).strip()
            system_prompt_parts.append(prompt_context)

        return "\n\n".join(system_prompt_parts).strip()

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

        plan = (response.content or "").strip()
        if not plan:
            return None

        self.history.add_message(
            Message(
                role="assistant",
                content=f"I have worked through this problem in-depth and my reasoning is summarized below.\n\n{plan}",
            )
        )

        logger.debug("Thought:\n" + plan)

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

        response = await reasoning_agent.arun(input)

        plan = (response.content or "").strip()
        if not plan:
            return None

        self.history.add_message(
            Message(
                role="assistant",
                content=f"I have worked through this problem in-depth and my reasoning is summarized below.\n\n{plan}",
            )
        )

        logger.debug("Thought:\n" + plan)

        return response

    def get_all_tools(self) -> list[BaseTool]:
        """All agent tools, including MCP tools and function tools."""
        tools = list(self.tools)
        for server in self.mcp_servers:
            tools += server.get_tools()
        return tools

    @tracer.tool(name="Tool Call")
    def run_tool(self, tool_call: ToolCall) -> Message:
        tool_execution = ToolResponse(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
        )

        tool_map = {tool.name: tool for tool in self.get_all_tools()}

        if tool_call.name not in tool_map.keys():
            logger.error(f"Tool {tool_call.name} does not exist")
            tool_execution.content = f"Tool {tool_call.name} does not exist"
            return tool_execution.to_message()

        if self.skills and tool_call.name == "load_skill":
            try:
                args = tool_call.arguments
                if not isinstance(args, dict):
                    args = json.loads(args) if args and args.strip() else {}
                skill_name = args.get("skill_name")
                logger.debug(f"[SKILL CALL] Loading skill '{skill_name}'")
                skill = next((s for s in self.skills if s.name == skill_name), None)
                if skill:
                    tool_execution.content = skill.to_prompt()
                    return tool_execution.to_message()
            except Exception as e:
                logger.error(f"Error loading skill '{tool_call.name}': {e}")

        try:
            tool = tool_map[tool_call.name]
            arguments = tool_call.arguments
            if not isinstance(arguments, dict):
                arguments = json.loads(arguments) if arguments and arguments.strip() else {}
            logger.info(f"[Tool Call] '{tool_call.name}'({arguments})")
            tool_execution.content = tool.run(**arguments)
            if tool_execution.content:
                return tool_execution.to_message()

        except Exception as e:
            logger.error(f"Error executing tool '{tool_call.name}': {e}")

        tool_execution.content = f"Error executing tool '{tool_call.name}'"
        return tool_execution.to_message()

    def execute_tool_calls(self, tool_calls: List[ToolCall]) -> List[Message]:
        results = []
        for tool_call in tool_calls:
            result = self.run_tool(tool_call)
            if result:
                results.append(result)

        return results

    async def aexecute_tool_calls(self, tool_calls: List[ToolCall]) -> List[Message]:
        tasks = []
        for tool_call in tool_calls:
            task = asyncio.create_task(asyncio.to_thread(self.run_tool, tool_call))
            tasks.append(task)

        results = await asyncio.gather(*tasks)

        return results

    @tracer.agent(name="Agent Run")
    def run(
        self,
        input: Union[str, List[Message], List[Dict[str, str]]],
        context: Optional[Union[str, Dict[str, str]]] = None,
    ) -> AgentResponse:
        # prepare messages and task
        messages = self.llm.input_to_message_list(input)
        task = messages[-1].content

        # init agent response
        agent_response = AgentResponse(agent_id=self.id)
        agent_response.events.append(
            AgentEventStarted(
                agent_id=self.id,
                run_id=agent_response.run_id,
            )
        )

        # history
        self._prepare_history(task=task, context=context)
        self.history.add_messages(messages)

        # add reasoning
        if self.reasoning_model:
            messages_for_reasoning_model = [m.to_dict() for m in self.history.get()]
            response = self.reason(messages_for_reasoning_model)
            agent_response.reasoning_content = response.reasoning_content
            agent_response.usage.add(response.usage)
            if response.reasoning_content:
                agent_response.events.append(
                    AgentEventThinking(
                        agent_id=self.id,
                        run_id=agent_response.run_id,
                        content=response.reasoning_content,
                    )
                )

        num_turns_available = self.max_turns

        while num_turns_available > 0:
            num_turns_available -= 1

            # format messages
            messages_for_model = [m.to_dict() for m in self.history.get()]

            # call llm and tool
            response = self.llm.invoke(input=messages_for_model)

            # usage
            agent_response.usage.add(response.usage)

            # native thinking from the main LLM call
            if response.thinking:
                if agent_response.reasoning_content:
                    agent_response.reasoning_content += "\n\n" + response.thinking
                else:
                    agent_response.reasoning_content = response.thinking
                agent_response.events.append(
                    AgentEventThinking(
                        agent_id=self.id,
                        run_id=agent_response.run_id,
                        content=response.thinking,
                    )
                )

            if response.content:
                agent_response.events.append(
                    AgentEventContent(
                        agent_id=self.id,
                        run_id=agent_response.run_id,
                        content=response.content,
                    )
                )

            # keep answer in memory (with provider-shaped thinking blocks for echo-back)
            tool_calls = [t.to_message_dict() for t in response.tool_calls]
            _message = Message(role="assistant", content=response.content, tool_calls=tool_calls)
            _message.thinking_parts = self.llm.get_thinking_parts(response)
            self.history.add_message(_message)

            # stop if not tool call
            if not response.tool_calls:
                content_to_check = response.parsed if response.parsed else response.content
                is_valid, parsed_data, error_msg = self._validate_final_response(content_to_check)

                if is_valid:
                    agent_response.content = response.content
                    agent_response.parsed = parsed_data
                    agent_response.messages.append(_message)
                    break
                else:
                    self.history.add_message(_message)
                    retry_message = Message(role="user", content=error_msg)
                    self.history.add_message(retry_message)
                    continue

            # handle tool calls
            if response.tool_calls and self.get_all_tools():
                for tool_call in response.tool_calls:
                    args = tool_call.arguments
                    if not isinstance(args, dict):
                        args = json.loads(args) if args and args.strip() else {}
                    agent_response.events.append(
                        AgentEventToolStarted(
                            agent_id=self.id,
                            run_id=agent_response.run_id,
                            tool_call_id=tool_call.id,
                            tool_name=tool_call.name,
                            tool_args=args,
                        )
                    )

                tool_messages = self.execute_tool_calls(tool_calls=response.tool_calls)
                for m in tool_messages:
                    agent_response.usage.tool_calls += 1
                    self.history.add_message(m)
                    agent_response.messages.append(m)
                    agent_response.events.append(
                        AgentEventToolCompleted(
                            agent_id=self.id,
                            run_id=agent_response.run_id,
                            tool_call_id=m.tool_call_id,
                            content=m.content,
                        )
                    )

            if self.tool_choice == "required":
                self.tool_choice = "auto"

        agent_response.events.append(
            AgentEventCompleted(
                agent_id=self.id,
                run_id=agent_response.run_id,
            )
        )

        agent_response.finish_reason = "stop"

        return agent_response

    @tracer.agent(name="Agent Arun")
    async def arun(
        self,
        input: Union[str, List[Message], List[Dict[str, str]]],
        context: Optional[Union[str, Dict[str, str]]] = None,
    ) -> AgentResponse:
        messages = self.llm.input_to_message_list(input)
        task = messages[-1].content

        agent_response = AgentResponse(agent_id=self.id)
        agent_response.events.append(
            AgentEventStarted(
                agent_id=self.id,
                run_id=agent_response.run_id,
            )
        )

        self._prepare_history(task=task, context=context)
        self.history.add_messages(messages)

        if self.reasoning_model:
            messages_for_reasoning_model = [m.to_dict() for m in self.history.get()]
            reasoning_agent_response = await self.areason(messages_for_reasoning_model)
            agent_response.reasoning_content = reasoning_agent_response.reasoning_content
            agent_response.usage.add(reasoning_agent_response.usage)
            if reasoning_agent_response.reasoning_content:
                agent_response.events.append(
                    AgentEventThinking(
                        agent_id=self.id,
                        run_id=agent_response.run_id,
                        content=reasoning_agent_response.reasoning_content,
                    )
                )

        num_turns_available = self.max_turns

        while num_turns_available > 0:
            num_turns_available -= 1

            messages_for_model = [m.to_dict() for m in self.history.get()]

            response = await self.llm.ainvoke(input=messages_for_model)

            # usage
            agent_response.usage.add(response.usage)

            if response.thinking:
                if agent_response.reasoning_content:
                    agent_response.reasoning_content += "\n\n" + response.thinking
                else:
                    agent_response.reasoning_content = response.thinking
                agent_response.events.append(
                    AgentEventThinking(
                        agent_id=self.id,
                        run_id=agent_response.run_id,
                        content=response.thinking,
                    )
                )

            if response.content:
                agent_response.events.append(
                    AgentEventContent(
                        agent_id=self.id,
                        run_id=agent_response.run_id,
                        content=response.content,
                    )
                )

            tool_calls = [t.to_message_dict() for t in response.tool_calls]
            _message = Message(role="assistant", content=response.content, tool_calls=tool_calls)
            _message.thinking_parts = self.llm.get_thinking_parts(response)
            self.history.add_message(_message)

            if not response.tool_calls:
                content_to_check = response.parsed if response.parsed else response.content
                is_valid, parsed_data, error_msg = self._validate_final_response(content_to_check)

                if is_valid:
                    agent_response.content = response.content
                    agent_response.parsed = parsed_data
                    agent_response.messages.append(_message)
                    break
                else:
                    self.history.add_message(_message)
                    retry_message = Message(role="user", content=error_msg)
                    self.history.add_message(retry_message)
                    continue

            # handle tool calls
            if response.tool_calls and self.get_all_tools():
                for tool_call in response.tool_calls:
                    args = tool_call.arguments
                    if not isinstance(args, dict):
                        args = json.loads(args) if args and args.strip() else {}
                    agent_response.events.append(
                        AgentEventToolStarted(
                            agent_id=self.id,
                            run_id=agent_response.run_id,
                            tool_call_id=tool_call.id,
                            tool_name=tool_call.name,
                            tool_args=args,
                        )
                    )

                tool_messages = await self.aexecute_tool_calls(tool_calls=response.tool_calls)
                for m in tool_messages:
                    agent_response.usage.tool_calls += 1
                    self.history.add_message(m)
                    agent_response.messages.append(m)
                    agent_response.events.append(
                        AgentEventToolCompleted(
                            agent_id=self.id,
                            run_id=agent_response.run_id,
                            tool_call_id=m.tool_call_id,
                            content=m.content,
                        )
                    )

            if self.tool_choice == "required":
                self.tool_choice = "auto"

        agent_response.events.append(
            AgentEventCompleted(
                agent_id=self.id,
                run_id=agent_response.run_id,
            )
        )

        agent_response.finish_reason = "stop"

        return agent_response

    @tracer.agent(name="Agent Stream")
    def run_stream(
        self,
        input: Union[str, List[Message], List[Dict[str, str]]],
        context: Optional[Union[str, Dict[str, str]]] = None,
    ) -> Iterator[AgentResponse]:
        messages = self.llm.input_to_message_list(input)
        task = messages[-1].content

        agent_response = AgentResponse(agent_id=self.id)
        event = AgentEventStarted(
            agent_id=self.id,
            run_id=agent_response.run_id,
        )
        agent_response.events.append(event)
        yield event

        self._prepare_history(task=task, context=context)
        self.history.add_messages(messages)

        if self.reasoning_model:
            messages_for_reasoning_model = [m.to_dict() for m in self.history.get()]
            reasoning_response = self.reason(messages_for_reasoning_model)
            agent_response.reasoning_content = reasoning_response.reasoning_content
            agent_response.usage.add(reasoning_response.usage)
            event = AgentEventThinking(
                agent_id=self.id, run_id=agent_response.run_id, content=reasoning_response.reasoning_content
            )
            agent_response.events.append(event)
            yield event

        num_turns_available = self.max_turns

        while num_turns_available > 0:
            num_turns_available -= 1

            messages_for_model = [m.to_dict() for m in self.history.get()]

            full_content = ""
            full_thinking = ""
            final_tool_calls: List[ToolCall] = []
            turn_usage = RequestUsage()
            # Accumulate raw thinking parts across the turn so we can re-derive
            # provider-shaped echo blocks (signatures etc.) at the end.
            turn_response = ModelResponse()

            for chunk in self.llm.stream(input=messages_for_model):
                if chunk.usage:
                    turn_usage += chunk.usage

                agent_response.content = None

                if chunk.thinking:
                    full_thinking += chunk.thinking
                    event = AgentEventThinking(
                        agent_id=self.id,
                        run_id=agent_response.run_id,
                        content=chunk.thinking,
                    )
                    agent_response.events.append(event)
                    yield event

                if chunk.content:
                    agent_response.content = chunk.content
                    full_content += chunk.content
                    event = AgentEventContent(agent_id=self.id, run_id=agent_response.run_id, content=chunk.content)
                    agent_response.events.append(event)
                    yield event

                if chunk.tool_calls:
                    final_tool_calls = chunk.tool_calls

                # Stash raw parts (incl. ThinkingContent with signatures) for echo-back
                turn_response.parts.extend(chunk.parts)

            agent_response.usage.add(turn_usage)

            if full_thinking:
                if agent_response.reasoning_content:
                    agent_response.reasoning_content += "\n\n" + full_thinking
                else:
                    agent_response.reasoning_content = full_thinking

            tool_calls_dicts = [t.to_message_dict() for t in final_tool_calls]
            _message = Message(role="assistant", content=full_content, tool_calls=tool_calls_dicts)
            _message.thinking_parts = self.llm.get_thinking_parts(turn_response)
            self.history.add_message(_message)

            if not final_tool_calls:
                agent_response.content = full_content
                agent_response.messages.append(_message)
                break

            if final_tool_calls and self.get_all_tools():
                for tool_call in final_tool_calls:
                    args = tool_call.arguments
                    if not isinstance(args, dict):
                        args = json.loads(args) if args and args.strip() else {}
                    event = AgentEventToolStarted(
                        agent_id=self.id,
                        run_id=agent_response.run_id,
                        tool_call_id=tool_call.id,
                        tool_name=tool_call.name,
                        tool_args=args,
                    )
                    agent_response.events.append(event)
                    yield event

                tool_messages = self.execute_tool_calls(tool_calls=final_tool_calls)
                for m in tool_messages:
                    agent_response.usage.tool_calls += 1
                    self.history.add_message(m)
                    agent_response.messages.append(m)
                    event = AgentEventToolCompleted(
                        agent_id=self.id, run_id=agent_response.run_id, tool_call_id=m.tool_call_id, content=m.content
                    )
                    agent_response.events.append(event)
                    yield event

            if self.tool_choice == "required":
                self.tool_choice = "auto"

        if self.response_model:
            try:
                agent_response.parsed = json.loads(full_content)
            except Exception as e:
                logger.error(f"Failed to parse JSON response: {e}")
                agent_response.parsed = {"error": f"Parse error: {str(e)}"}

        event = AgentEventCompleted(
            agent_id=self.id,
            run_id=agent_response.run_id,
        )
        agent_response.events.append(event)
        yield event

        agent_response.finish_reason = "stop"

        yield agent_response

    @tracer.agent(name="Agent Astream")
    async def arun_stream(
        self,
        input: Union[str, List[Message], List[Dict[str, str]]],
        context: Optional[Union[str, Dict[str, str]]] = None,
    ) -> AsyncIterator[AgentResponse]:
        messages = self.llm.input_to_message_list(input)
        task = messages[-1].content

        agent_response = AgentResponse(agent_id=self.id)
        event = AgentEventStarted(
            agent_id=self.id,
            run_id=agent_response.run_id,
        )
        agent_response.events.append(event)
        yield event

        self._prepare_history(task=task, context=context)
        self.history.add_messages(messages)

        if self.reasoning_model:
            messages_for_reasoning_model = [m.to_dict() for m in self.history.get()]
            reasoning_response = await self.areason(messages_for_reasoning_model)
            agent_response.reasoning_content = reasoning_response.reasoning_content
            agent_response.usage.add(reasoning_response.usage)
            event = AgentEventThinking(
                agent_id=self.id, run_id=agent_response.run_id, content=reasoning_response.reasoning_content
            )
            agent_response.events.append(event)
            yield event

        num_turns_available = self.max_turns

        while num_turns_available > 0:
            num_turns_available -= 1

            messages_for_model = [m.to_dict() for m in self.history.get()]

            full_content = ""
            full_thinking = ""
            final_tool_calls: List[ToolCall] = []
            turn_usage = RequestUsage()
            turn_response = ModelResponse()

            async for chunk in self.llm.astream(input=messages_for_model):
                if chunk.usage:
                    turn_usage += chunk.usage

                agent_response.content = None

                if chunk.thinking:
                    full_thinking += chunk.thinking
                    event = AgentEventThinking(
                        agent_id=self.id,
                        run_id=agent_response.run_id,
                        content=chunk.thinking,
                    )
                    agent_response.events.append(event)
                    yield event

                if chunk.content:
                    agent_response.content = chunk.content
                    full_content += chunk.content
                    event = AgentEventContent(agent_id=self.id, run_id=agent_response.run_id, content=chunk.content)
                    agent_response.events.append(event)
                    yield event

                if chunk.tool_calls:
                    final_tool_calls = chunk.tool_calls

                turn_response.parts.extend(chunk.parts)

            agent_response.usage.add(turn_usage)

            if full_thinking:
                if agent_response.reasoning_content:
                    agent_response.reasoning_content += "\n\n" + full_thinking
                else:
                    agent_response.reasoning_content = full_thinking

            tool_calls_dicts = [t.to_message_dict() for t in final_tool_calls]
            _message = Message(role="assistant", content=full_content, tool_calls=tool_calls_dicts)
            _message.thinking_parts = self.llm.get_thinking_parts(turn_response)
            self.history.add_message(_message)

            if not final_tool_calls:
                agent_response.content = full_content
                agent_response.messages.append(_message)
                break

            if final_tool_calls and self.get_all_tools():
                for tool_call in final_tool_calls:
                    args = tool_call.arguments
                    if not isinstance(args, dict):
                        args = json.loads(args) if args and args.strip() else {}
                    event = AgentEventToolStarted(
                        agent_id=self.id,
                        run_id=agent_response.run_id,
                        tool_call_id=tool_call.id,
                        tool_name=tool_call.name,
                        tool_args=args,
                    )
                    agent_response.events.append(event)
                    yield event

                tool_messages = await self.aexecute_tool_calls(tool_calls=final_tool_calls)
                for m in tool_messages:
                    agent_response.usage.tool_calls += 1
                    self.history.add_message(m)
                    agent_response.messages.append(m)
                    event = AgentEventToolCompleted(
                        agent_id=self.id, run_id=agent_response.run_id, tool_call_id=m.tool_call_id, content=m.content
                    )
                    agent_response.events.append(event)
                    yield event

            if self.tool_choice == "required":
                self.tool_choice = "auto"

        if self.response_model:
            try:
                agent_response.parsed = json.loads(full_content)
            except Exception as e:
                logger.error(f"Failed to parse JSON response: {e}")
                agent_response.parsed = {"error": f"Parse error: {str(e)}"}

        event = AgentEventCompleted(
            agent_id=self.id,
            run_id=agent_response.run_id,
        )
        agent_response.events.append(event)
        yield event

        agent_response.finish_reason = "stop"

        yield agent_response
