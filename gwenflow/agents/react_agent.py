import json
import re
import uuid
from typing import Dict, Iterator, List, Optional, Union

from pydantic import BaseModel

from gwenflow.agents.agent import DEFAULT_MAX_TURNS, Agent
from gwenflow.logger import logger
from gwenflow.types import AgentResponse, ItemHelpers, Message, ToolCall, Usage

PROMPT_TOOLS = """\
You have access to the following tools. Only use these tools.

```json
{tools}
```
"""

PROMPT_REACT = """\
Please answer in the following format:

```
Thought: analyze the problem, plan the next action.
Action: tool name (one of {tool_names}) if using a tool.
Action Input: the input to the tool, in a JSON format representing the kwargs (e.g. {{"input": "hello world", "num_beams": 5}})
```

Please ALWAYS start with a Thought.

NEVER surround your response with markdown code markers. You may use code markers within your response if you need to.

Please use a valid JSON format for the Action Input. Do NOT do this {{'input': 'hello world', 'num_beams': 5}}.

If this format is used, the tool will respond in the following format:

```
Observation: tool response
```

You should keep repeating the above format till you have enough information to answer the question without using any more tools. At that point, you MUST respond in one of the following two formats:

```
Thought: I can answer without using any more tools. I'll use the user's language to answer
Final Answer: [your answer here (In the same language as the user's question)]
```

```
Thought: I cannot answer the question with the provided tools.
Final Answer: [your answer here (In the same language as the user's question)]
```

Question: {query}"""


SPECIAL_TOK_THOUGHT = "Thought:"
SPECIAL_TOK_ACTION = "Action:"
SPECIAL_TOK_FINAL_ANSWER = "Final Answer:"


class ReactAgentAction(BaseModel):
    thought: Optional[str] = None
    action: str
    action_input: str

    def get_tool_call(self) -> ToolCall:
        return ToolCall(
            id=str(uuid.uuid4()),
            function=self.action,
            arguments=json.loads(self.action_input),
        )


class ReactAgentFinish(BaseModel):
    final_answer: str


class ReactMessageParser(BaseModel):
    @classmethod
    def parse(cls, text: str) -> Union[ReactAgentAction, ReactAgentFinish]:
        includes_answer = SPECIAL_TOK_FINAL_ANSWER in text
        regex = r"Action\s*\d*\s*:[\s]*(.*?)[\s]*Action\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        action_match = re.search(regex, text, re.DOTALL)
        if action_match:
            if includes_answer:
                msg = f"Parsing LLM output produced both a final answer and a parse-able action: {text}"
                raise Exception(msg)
            action = action_match.group(1).strip()
            action_input = action_match.group(2)
            tool_input = action_input.strip(" ")
            tool_input = tool_input.strip('"')

            i = text.rfind(SPECIAL_TOK_ACTION)
            thought = text[:i].strip()
            if thought.startswith(SPECIAL_TOK_THOUGHT):
                thought = thought.split(SPECIAL_TOK_THOUGHT)[-1].strip()

            return ReactAgentAction(action=action, action_input=tool_input, thought=thought)

        if SPECIAL_TOK_FINAL_ANSWER in text:
            final_answer = text.split(SPECIAL_TOK_FINAL_ANSWER)[-1].strip()
            return ReactAgentFinish(final_answer=final_answer)

        msg = f"Could not parse: `{text}`"
        raise Exception(msg)


class ReactAgent(Agent):
    def _prepend_react_prompt(self, messages: List[Message]) -> List[Message]:
        tool_names = ",".join(t.name for t in self.tools)
        tool_descs = [t.to_openai_chat_completion()["function"] for t in self.tools]
        prompt_tools = PROMPT_TOOLS.format(tools=json.dumps(tool_descs, indent=2, ensure_ascii=False))

        messages[-1].content = (
            prompt_tools
            + "\n"
            + PROMPT_REACT.format(
                tool_names=tool_names,
                query=messages[-1].content,
            )
        )

        return messages

    def run(
        self,
        input: Union[str, List[Message], List[Dict[str, str]]],
        context: Optional[Union[str, Dict[str, str]]] = None,
    ) -> AgentResponse:
        # prepare messages and task
        messages = ItemHelpers.input_to_message_list(input)
        task = messages[-1].content

        # add react prompt to the last message
        self.llm.tool_type = "react"
        messages = self._prepend_react_prompt(messages)

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

        num_turns_available = DEFAULT_MAX_TURNS

        while num_turns_available > 0:
            num_turns_available -= 1

            # format messages
            messages_for_model = [m.to_dict() for m in self.history.get()]

            # call llm and tool
            response = self.llm.invoke(input=messages_for_model)

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

            parsed_message = ReactMessageParser.parse(response.choices[0].message.content)

            # keep thought
            self.history.add_message(Message(role="assistant", content=response.choices[0].message.content))
            agent_response.messages.append(Message(role="assistant", content=response.choices[0].message.content))

            # stop if not tool call
            if isinstance(parsed_message, ReactAgentFinish):
                agent_response.content = parsed_message.final_answer
                agent_response.messages.append(Message(**response.choices[0].message.model_dump()))
                break

            # thinking
            agent_response.thinking = parsed_message.thought
            logger.debug(parsed_message.thought)

            # handle tool calls
            tool_message = self.run_tool(parsed_message.get_tool_call())
            self.history.add_message(Message(role="user", content=f"Observation: {tool_message.content}"))
            agent_response.messages.append(Message(role="user", content=f"Observation: {tool_message.content}"))

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
        messages = ItemHelpers.input_to_message_list(input)
        task = messages[-1].content

        # add react prompt to the last message
        self.llm.tool_type = "react"
        messages = self._prepend_react_prompt(messages)

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

        num_turns_available = DEFAULT_MAX_TURNS

        while num_turns_available > 0:
            num_turns_available -= 1

            # format messages
            messages_for_model = [m.to_dict() for m in self.history.get()]

            # call llm and tool
            last_delta_message = None
            output = ""

            for chunk in self.llm.stream(input=messages_for_model):
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
                if not delta.content:
                    continue

                output += delta.content

                special_func_token = "Action: "
                special_args_token = "Action Input: "
                special_thought_token = "Thought: "
                special_final_token = "Final Answer: "

                a = output.rfind(special_func_token)  # action
                i = output.rfind(special_args_token)  # iaction nput
                t = output.rfind(special_thought_token)  # thought
                f = output.rfind(special_final_token)  # final

                last_delta_message = None
                if max([a, i, t, f]) == t:
                    agent_response.content = None
                    if last_delta_message != "thought":
                        last_delta_message = "thought"
                        agent_response.thinking = None
                    else:
                        agent_response.thinking = delta.content
                elif max([a, i, t, f]) == f:
                    agent_response.thinking = None
                    if last_delta_message != "final":
                        last_delta_message = "final"
                        agent_response.content = None
                    else:
                        agent_response.content = delta.content

                yield agent_response

            parsed_message = ReactMessageParser.parse(output)

            # keep thought
            self.history.add_message(Message(role="assistant", content=output))
            agent_response.messages.append(Message(role="assistant", content=output))

            # stop if no tool call
            if isinstance(parsed_message, ReactAgentFinish):
                agent_response.content = parsed_message.final_answer
                agent_response.messages.append(Message(role="assistant", content=output))
                break

            # thinking
            agent_response.thinking = parsed_message.thought
            if agent_response.thinking:
                logger.debug(parsed_message.thought)
                yield agent_response

            # handle tool calls
            observation = self.run_tool(parsed_message.get_tool_call())
            self.history.add_message(Message(role="user", content=f"Observation: {observation.content}"))
            agent_response.messages.append(Message(role="user", content=f"Observation: {observation.content}"))

        # format response
        if self.response_model:
            agent_response.content = json.loads(agent_response.content)

        agent_response.finish_reason = "stop"

        yield agent_response
