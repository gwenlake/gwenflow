import logging
import json
from typing import List, Callable, Union, Any
from collections import defaultdict

from gwenflow.types import ChatCompletionMessageToolCall, Function
from gwenflow.agents.agent import Agent, Response
from gwenflow.agents.prompts import CONTEXT, EXPECTED_OUTPUT
from gwenflow.agents.utils import merge_chunk


MAX_LOOPS = 10


logger = logging.getLogger(__name__)


class Task:

    def __init__(self, *, description: str, expected_output: str = None, agent: Agent):

        self.description = description
        self.expected_output = expected_output
        self.agent = agent

    def prompt(self, context: str = None) -> str:
        """Prompt the task.

        Returns:
            Prompt of the task.
        """
        _prompt = [self.description]
        _prompt.append(EXPECTED_OUTPUT.format(expected_output=self.expected_output))
        if context:
            _prompt.append(CONTEXT.format(context=context))
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

    def run(self, context: str = None, context_variables: dict = {}) -> str:
        
        task_prompt  = self.prompt(context)
        active_agent = self.agent
        
        num_loops = 1
        while active_agent and num_loops < MAX_LOOPS:

            response = active_agent.execute_task(task_prompt, context_variables=context_variables)
            
            # task done
            if isinstance(response, Response):
                response = response.output
                break

            # task transfered to another agent
            elif isinstance(response, Agent):
                active_agent = response

            num_loops += 1

        return response
