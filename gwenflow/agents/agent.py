
import os
import shutil
import subprocess
from typing import List, Callable, Union, Optional, Any, Dict
from collections import defaultdict
from pydantic import BaseModel
import logging
import json

from gwenflow.types import ChatCompletionMessage, ChatCompletionMessageToolCall, Function
from gwenflow.tools import Tool
from gwenflow.agents.types import (
    AgentTool,
    Response,
    Result,
)

MAX_TURNS = 10
__CTX_VARS_NAME__ = "context_variables"


logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)


class Agent(BaseModel):

    role: str
    instructions: Union[str, Callable[[], str]] = "You are a helpful agent."
    llm: Any
    tools: List[Tool] = []
    tool_choice: str = None
    parallel_tool_calls: bool = True
    messages: List[Dict[str, str]] = [] # TODO: to add to the object (move from tasks)

    # stockés dans prompts.py
    # def _format_prompt(self, prompt: str, inputs: Dict[str, str]) -> str:
    #     prompt = prompt.replace("{input}", inputs["input"])
    #     prompt = prompt.replace("{tool_names}", inputs["tool_names"])
    #     prompt = prompt.replace("{tools}", inputs["tools"])
    #     return prompt

    def handle_function_result(self, result) -> Result:
        match result:
            case Result() as result:
                return result

            case Agent() as agent:
                return Result(
                    value=json.dumps({"assistant": self.role}),
                    agent=agent,
                )
            case _:
                try:
                    return Result(value=str(result))
                except Exception as e:
                    error_message = f"Failed to cast response to string: {result}. Make sure agent functions return a string or Result object. Error: {str(e)}"
                    logger.error(error_message)
                    raise TypeError(error_message)
                
    def handle_tool_calls(
        self,
        tool_calls: List[ChatCompletionMessageToolCall],
        tools: List[Tool],
        context_variables: dict,
    ) -> Response:
        
        # function_map = {f.__name__: f for f in tools}
        tool_map = {tool.name: tool for tool in tools}

        partial_response = Response(messages=[], agent=None, context_variables={})

        for tool_call in tool_calls:
            name = tool_call.function.name
            # handle missing tool case, skip to next tool
            if name not in tool_map:
                logger.debug(f"Tool {name} not found in function map.")
                partial_response.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "tool_name": name,
                        "content": f"Error: Tool {name} not found.",
                    }
                )
                continue
            args = json.loads(tool_call.function.arguments)
            logger.debug(f"Processing tool call: {name} with arguments {args}")

            # tool = tool_map[name]
            # pass context_variables to agent functions
            # TODO: adjust -> var names?
            # if __CTX_VARS_NAME__ in func.__code__.co_varnames:
            #     args[__CTX_VARS_NAME__] = context_variables

            raw_result = tool_map[name].run(**args)

            result: Result = self.handle_function_result(raw_result)
            partial_response.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "tool_name": name,
                    "content": result.value,
                }
            )
            partial_response.context_variables.update(result.context_variables)
            if result.agent:
                partial_response.agent = result.agent

        return partial_response

    def invoke(self, stream: bool = False):

        # tools in OpenAI json format
        tools = [tool.openai_schema for tool in self.tools]

        # hide context_variables from model
        for tool in tools:
            params = tool["function"]["parameters"]
            params["properties"].pop(__CTX_VARS_NAME__, None)
            if __CTX_VARS_NAME__ in params["required"]:
                params["required"].remove(__CTX_VARS_NAME__)

        params = {
            "messages": self.messages,
            "tools": tools or None,
            "tool_choice": self.tool_choice,
            "parse_response": False,
        }

        if tools:
            params["parallel_tool_calls"] = self.parallel_tool_calls

        if stream:
            params["stream"] = True
            return self.llm.stream(**params)
        
        return self.llm.invoke(**params)

    def execute_task(
        self,
        task: str,
        context: Optional[str] = None,
        context_variables: Optional[dict] = None,
        # tools: Optional[List[BaseTool]] = None,
    ) -> str:

        # format instructions
        context_variables = defaultdict(str, context_variables)
        instructions = (
            self.instructions(context_variables)
            if callable(self.instructions)
            else self.instructions
        )

        # TODO: add context to task prompt if any

        # prepare messages
        self.messages = [
            { "role": "system", "content": instructions },
            { "role": "user",   "content": task },
        ]

        # global loop
        init_len = len(self.messages)
        while len(self.messages) - init_len < MAX_TURNS:

            completion = self.invoke()

            message = completion.choices[0].message
            message.sender = self.role

            self.messages.append(json.loads(message.model_dump_json()))  # to avoid OpenAI types (?)

            # check if done
            if not message.tool_calls:
                logging.debug("Task done.")
                return Response(
                    output=self.messages[-1]["content"],
                    messages=self.messages[init_len:],
                    agent=self,
                    context_variables=context_variables,
                )

            # handle function calls, updating context_variables
            partial_response = self.handle_tool_calls(message.tool_calls, self.tools, context_variables)
            self.messages.extend(partial_response.messages)
            context_variables.update(partial_response.context_variables)

            # switching agent?
            if partial_response.agent:
                logging.debug(f"Task transfered to Agent[{ partial_response.agent.name }].")
                return partial_response.agent

        logging.debug(f"Task failed")
        return None

    def _validate_docker_installation(self) -> None:
        """Check if Docker is installed and running."""
        if not shutil.which("docker"):
            raise RuntimeError(
                f"Docker is not installed. Please install Docker to use code execution with agent: {self.role}"
            )

        try:
            subprocess.run(
                ["docker", "info"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError:
            raise RuntimeError(
                f"Docker is not running. Please start Docker to use code execution with agent: {self.role}"
            )