import logging
import json
from typing import List, Callable, Union, Any
from collections import defaultdict

from gwenflow.types import ChatCompletionMessageToolCall, Function
from gwenflow.agents.agent import Agent, Response
from gwenflow.agents.prompts import TASK, EXPECTED_OUTPUT
from gwenflow.agents.utils import merge_chunk


logger = logging.getLogger(__name__)


class Task:

    def __init__(self, *, description: str, expected_output: str = None, agent: Agent):

        self.description = description
        self.expected_output = expected_output
        self.agent = agent

    def prompt(self) -> str:
        """Prompt the task.

        Returns:
            Prompt of the task.
        """
        _prompt = [self.description]
        # _prompt.append( TASK.format(description=self.description) )
        _prompt.append( EXPECTED_OUTPUT.format(expected_output=self.expected_output) )
        return "\n\n".join(_prompt).strip()

    def stream(
        self,
        context_variables: dict = {},
        max_turns: int = float("inf"),
    ):

        task_prompt = self.prompt()
        messages = [{"role": "user", "content": task_prompt}]

        init_len = len(messages)

        active_agent = self.agent

        while len(messages) - init_len < max_turns:

            message = {
                "content": "",
                "sender": self.agent.role,
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

            # get completion with current history, agent
            completion = active_agent.invoke(
                messages=messages,
                context_variables=context_variables,
                stream=True,
            )

            yield {"delim": "start"}
            for chunk in completion:
                delta = json.loads(chunk.choices[0].delta.json())
                if delta["role"] == "assistant":
                    delta["sender"] = active_agent.name
                yield delta
                delta.pop("role", None)
                delta.pop("sender", None)
                merge_chunk(message, delta)
            yield {"delim": "end"}

            message["tool_calls"] = list(
                message.get("tool_calls", {}).values())
            if not message["tool_calls"]:
                message["tool_calls"] = None
            logging.debug("Received completion:", message)
            messages.append(message)

            if not message["tool_calls"]:
                logging.debug("Ending turn.")
                break

            # convert tool_calls to objects
            tool_calls = []
            for tool_call in message["tool_calls"]:
                function = Function(
                    arguments=tool_call["function"]["arguments"],
                    name=tool_call["function"]["name"],
                )
                tool_call_object = ChatCompletionMessageToolCall(
                    id=tool_call["id"], function=function, type=tool_call["type"]
                )
                tool_calls.append(tool_call_object)

            # handle function calls, updating context_variables, and switching agents
            partial_response = self.handle_tool_calls(tool_calls, active_agent.tools, context_variables)
            messages.extend(partial_response.messages)
            context_variables.update(partial_response.context_variables)
            if partial_response.agent:
                active_agent = partial_response.agent

        yield {
            "response": Response(
                output=messages[init_len]["content"],
                messages=messages[init_len:],
                agent=active_agent,
                context_variables=context_variables,
            )
        }

    def run(
        self,
        context_variables: dict = {},
        max_turns: int = float("inf"),
    ) -> Response:
        
        # prepare messages
        task_prompt = self.prompt()
        messages = [{"role": "user", "content": task_prompt}]

        init_len = len(messages)

        # current agent
        active_agent = self.agent

        # global loop
        while len(messages) - init_len < max_turns and active_agent:

            completion = active_agent.invoke(messages=messages, context_variables=context_variables)

            message = completion.choices[0].message
            message.sender = active_agent.role
            messages.append(json.loads(message.model_dump_json()))  # to avoid OpenAI types (?)

            # check if done
            if not message.tool_calls:
                logging.debug("Ending turn.")
                break

            # handle function calls, updating context_variables
            partial_response = active_agent.handle_tool_calls(message.tool_calls, active_agent.tools, context_variables)
            messages.extend(partial_response.messages)
            context_variables.update(partial_response.context_variables)

            # switching agent?
            if partial_response.agent:
                active_agent = partial_response.agent

        return Response(
            output=messages[-1]["content"],
            messages=messages[init_len:],
            agent=active_agent,
            context_variables=context_variables,
        )