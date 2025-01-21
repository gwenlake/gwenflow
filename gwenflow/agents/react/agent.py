import json
from typing import Union, Optional, Any, Dict, Iterator
from collections import defaultdict

from gwenflow.types import ChatCompletionMessage
from gwenflow.agents.types import AgentResponse
from gwenflow.agents.agent import Agent
from gwenflow.agents.react.types import ActionReasoningStep
from gwenflow.agents.react.parser import parse_reasoning_step
from gwenflow.agents.react.prompts import PROMPT_TASK_REACT
from gwenflow.agents.utils import merge_chunk
from gwenflow.utils import logger


MAX_TURNS = float('inf')


class ReActAgent(Agent):

    is_react: bool = True

    def get_user_message(self, task: Optional[str] = None, context: Optional[Any] = None):
        """Return the user message for the Agent."""

        if not task:
            raise ValueError("A ReActAgent always needs a task!")

        prompt = ""

        if context:
            prompt += self.format_context(context)

        tool_names = ",".join(self.get_tool_names())
        prompt += PROMPT_TASK_REACT.format(task=task, tool_names=tool_names)

        return { "role": "user", "content": prompt }


    def handle_tool_call(
        self,
        reasoning_step: ActionReasoningStep,
    ) -> Dict:
        
        tool_map = self.get_tools_map()

        # handle missing tool case, skip to next tool
        if reasoning_step.action not in tool_map:
            logger.error(f"Unknown tool {reasoning_step.action}, should be instead one of { tool_map.keys() }.")
            return {
                    "role": "assistant",
                    "content": f"Error: Tool {reasoning_step.action} not found.",
            }

        arguments = json.loads(reasoning_step.action_input)
        observation = self.execute_tool_call(reasoning_step.action, arguments)
                
        return {
            "role": "assistant",
            "content": f"Observation: {observation}",
        }
    
    def invoke(self, messages: list, stream: bool = False) ->  Union[Any, Iterator[Any]]:

        params = {
            "messages": messages,
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
            self.history.add_message(user_message)

        # global loop
        response = ""
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

            # parse response
            logger.info(completion.choices[0].message.content)
            reasoning_step = parse_reasoning_step(completion.choices[0].message.content)
            if reasoning_step.is_done:
                logger.debug("Task done.")
                response = reasoning_step.response
                break

            # handle tool calls and switching agents
            observation = self.handle_tool_call(reasoning_step)
            messages_for_model.append(observation)

        content = messages_for_model[-1]["content"]
        if self.response_model:
            content = json.loads(content)

        yield AgentResponse(
            content=response,
            messages=messages_for_model[init_len:],
            agent=self,
            tools=self.tools,
        )

