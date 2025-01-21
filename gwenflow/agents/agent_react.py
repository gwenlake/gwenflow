
import uuid
import re
import json
from json_repair import repair_json
from typing import List, Callable, Union, Optional, Any, Dict, Iterator, Literal, Sequence, overload, Type
from collections import defaultdict
from pydantic import BaseModel, model_validator, field_validator, Field, UUID4
from datetime import datetime

from gwenflow.llms import ChatOpenAI
from gwenflow.types import ChatCompletionMessage, ChatCompletionMessageToolCall
from gwenflow.tools import BaseTool
from gwenflow.memory import ChatMemoryBuffer
from gwenflow.knowledge import Knowledge
from gwenflow.agents.types import AgentResponse
from gwenflow.agents.utils import merge_chunk
from gwenflow.utils import logger
from gwenflow.agents.agent import Agent
from gwenflow.agents.types import ReActAgentAction, ReActAgentFinish
from gwenflow.agents.prompts import PROMPT_TASK_REACT


MAX_TURNS = float('inf')

FINAL_ANSWER = "Final Answer:"

def _extract_thought(text: str) -> str:
    regex = r"(.*?)(?:\n\nAction|\n\nFinal Answer)"
    thought_match = re.search(regex, text, re.DOTALL)
    if thought_match:
        return thought_match.group(1).strip()
    return ""

def _clean_action(text: str) -> str:
    """Clean action string by removing non-essential formatting characters."""
    return re.sub(r"^\s*\*+\s*|\s*\*+\s*$", "", text).strip()

def _safe_repair_json(tool_input: str) -> str:
    UNABLE_TO_REPAIR_JSON_RESULTS = ['""', "{}"]
    if tool_input.startswith("[") and tool_input.endswith("]"):
        return tool_input
    tool_input = tool_input.replace('"""', '"')
    result = repair_json(tool_input)
    if result in UNABLE_TO_REPAIR_JSON_RESULTS:
        return tool_input
    return str(result)
    
def _parse_response(text: str) -> Union[ReActAgentAction, ReActAgentFinish]:

    thought = _extract_thought(text)
    includes_answer = "Final Answer:" in text
    regex = (
        r"Action\s*\d*\s*:[\s]*(.*?)[\s]*Action\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
    )
    action_match = re.search(regex, text, re.DOTALL)

    if action_match:

        if includes_answer:
            raise ValueError("Error while trying to perform Action and give a Final Answer at the same time!")

        action = action_match.group(1)
        action = _clean_action(action)
        action_input = action_match.group(2).strip()

        tool_input = action_input.strip(" ").strip('"')
        tool_input = _safe_repair_json(tool_input)

        return ReActAgentAction(thought=thought, action=action, action_input=tool_input, text=text)

    elif includes_answer:
        final_answer = text.split(FINAL_ANSWER)[-1].strip()
        return ReActAgentFinish(thought=thought, final_answer=final_answer, text=text)

    if not re.search(r"Action\s*\d*\s*:[\s]*(.*?)", text, re.DOTALL):
        raise ValueError("Missing Action after Thought!")

    elif not re.search(r"[\s]*Action\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)", text, re.DOTALL):
        raise ValueError("Missing Action Input after Action!")
    else:
        raise ValueError("Sorry, I didn't use the right tool format.!")


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
        agent_action: ReActAgentAction,
    ) -> Dict:
        
        tool_map = self.get_tools_map()

        # handle missing tool case, skip to next tool
        if agent_action.action not in tool_map:
            logger.error(f"Unknown tool {agent_action.action}, should be instead one of { tool_map.keys() }.")
            return {
                    "role": "assistant",
                    "content": f"Error: Tool {agent_action.action} not found.",
            }

        arguments = json.loads(agent_action.action_input)
        observation = self.execute_tool_call(agent_action.action, arguments)
        
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
        final_answer = ""
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
            parsed_response = _parse_response(completion.choices[0].message.content)
            if isinstance(parsed_response, ReActAgentFinish):
                logger.debug("Task done.")
                final_answer = parsed_response.final_answer
                break

            # handle tool calls and switching agents
            observation = self.handle_tool_call(parsed_response)
            messages_for_model.append(observation)

        content = messages_for_model[-1]["content"]
        if self.response_model:
            content = json.loads(content)

        yield AgentResponse(
            content=final_answer,
            messages=messages_for_model[init_len:],
            agent=self,
            tools=self.tools,
        )

