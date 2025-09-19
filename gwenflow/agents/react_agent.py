import json
import re
import asyncio
import uuid

from typing import List, Union, Optional, Any, Dict, Iterator, Literal, Tuple
from pydantic import BaseModel, model_validator, field_validator, Field, ConfigDict, UUID4

from gwenflow.logger import logger
from gwenflow.types import Usage, Message, AgentResponse, ResponseOutputItem, ItemHelpers
from gwenflow.agents.agent import Agent, DEFAULT_MAX_TURNS

from openai.types.chat import ChatCompletionMessageToolCall


TOOL_DESC = """<tool>
Name: {name}
Description: {description}
Arguments: {parameters}
</tool>"""

PROMPT_REACT = """\
## Tools
You have access to the following tools. Only use these tools.

<tools>
{tool_descs}
</tools>

## Format

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

## Task
Question: {query}"""


class ReactMessage(BaseModel):
    action: Optional[str] = None
    action_input: Optional[str] = None
    thought: Optional[str] = None
    final_answer: Optional[str] = None

    def get_tool_call(self):
        return dict(
            id=str(uuid.uuid4()),
            name=self.action,
            type="function",
            function={
                "name": self.action,
                "arguments": self.action_input
            }
        )
    
class ReactMessageParser(BaseModel):

    @classmethod
    def parse(self, text: str) -> ReactMessage:
        special_func_token = 'Action: '
        special_args_token = 'Action Input: '
        special_obs_token = 'Observation: '
        special_final_token = 'Final Answer: '
        func_name, func_args, final = None, None, None
        i = text.rfind(special_func_token)
        j = text.rfind(special_args_token)
        k = text.rfind(special_obs_token)
        if 0 <= i < j:  # If the text has `Action` and `Action input`,
            if k < j:  # but does not contain `Observation`,
                # then it is likely that `Observation` is ommited by the LLM,
                # because the output text may have discarded the stop word.
                text = text.rstrip() + special_obs_token  # Add it back.
            k = text.rfind(special_obs_token)
            func_name = text[i + len(special_func_token):j].strip()
            func_args = text[j + len(special_args_token):k].strip()
            text = text[:i]  # Return the response before tool call, i.e., `Thought`
        elif text.rfind(special_final_token):
            f = text.rfind(special_final_token)
            final = text[f + len(special_args_token):].strip()

        return ReactMessage(action=func_name, action_input=func_args, thought=text, final_answer=final)


class ReactAgent(Agent):
    
    def _prepend_react_prompt(self, messages: List[Message]) -> List[Message]:
        tool_descs = []
        for tool in self.tools:
            tool_descs.append(
                TOOL_DESC.format(
                    name=tool.name,
                    description=tool.description,
                    parameters=json.dumps(tool.params_json_schema)
                )
            )
        tool_descs = '\n'.join(tool_descs)
        tool_names = ','.join(tool.name for tool in self.tools)
        messages[-1].content = PROMPT_REACT.format(
            tool_descs=tool_descs,
            tool_names=tool_names,
            query=messages[-1].content,
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

            react_message = ReactMessageParser.parse(response.choices[0].message.content)

            # stop if not tool call
            if not react_message.action or not self.get_all_tools():
                agent_response.content = react_message.final_answer
                agent_response.output.append(Message(**response.choices[0].message.model_dump()))
                break
            
            # thinking
            agent_response.thinking = react_message.thought
            logger.debug(react_message.thought)

            # handle tool calls
            tool_message = self.run_tool(react_message.get_tool_call())
            text_message = f"Thought: {react_message.thought}\nAction: {react_message.action}\nAction Input: {react_message.action_input}\nObservation: { tool_message.content }"
            self.history.add_message(Message(role="assistant", content=text_message))
            agent_response.output.append(Message(role="assistant", content=text_message))
        
        # format response
        if self.response_model:
            agent_response.content = json.loads(agent_response.content)

        # keep sources
        for output in agent_response.output:
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

                special_func_token = 'Action: '
                special_args_token = 'Action Input: '
                special_thought_token = 'Thought: '
                special_final_token = 'Final Answer: '

                a = output.rfind(special_func_token) # action
                i = output.rfind(special_args_token) # iaction nput
                t = output.rfind(special_thought_token) # thought
                f = output.rfind(special_final_token) # final
                
                last_delta_message = None
                if max([a,i,t,f]) == t:
                    agent_response.content = None
                    if last_delta_message != "thought":
                        last_delta_message = "thought"
                        agent_response.thinking = None
                    else:
                        agent_response.thinking = delta.content
                elif max([a,i,t,f]) == f:
                    agent_response.thinking = None
                    if last_delta_message != "final":
                        last_delta_message = "final"
                        agent_response.content = None
                    else:
                        agent_response.content = delta.content
                
                yield agent_response


            react_message = ReactMessageParser.parse(output)

            # stop if no tool call
            if not react_message.action or not self.get_all_tools():
                agent_response.content = react_message.final_answer
                agent_response.output.append(Message(role="assistant", content=output))
                break

            # thinking
            agent_response.thinking = react_message.thought
            if agent_response.thinking:
                logger.debug(react_message.thought)
                yield agent_response

            # handle tool calls
            tool_message = self.run_tool(react_message.get_tool_call())
            text_message = f"Thought: {react_message.thought}\nAction: {react_message.action}\nAction Input: {react_message.action_input}\nObservation: { tool_message.content }"
            self.history.add_message(Message(role="assistant", content=text_message))
            agent_response.output.append(Message(role="assistant", content=text_message))
        
        # format response
        if self.response_model:
            agent_response.content = json.loads(agent_response.content)

        # keep sources
        for output in agent_response.output:
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

        yield agent_response
