
from typing import List, Callable, Union, Optional, Any, Dict
from collections import defaultdict
from pydantic import BaseModel
import logging
import json

from gwenflow.types import ChatCompletionMessage, ChatCompletionMessageToolCall, Function
from gwenflow.agents.utils import function_to_json
from gwenflow.agents.types import (
    AgentTool,
    Response,
    Result,
)


__CTX_VARS_NAME__ = "context_variables"


logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)


class Agent(BaseModel):

    role: str
    instructions: Union[str, Callable[[], str]] = "You are a helpful agent."
    llm: Any
    tools: List[AgentTool] = []
    tool_choice: str = None
    parallel_tool_calls: bool = True
    messages: List[Dict[str, str]] = [] # TODO: to add to the object (move from tasks)
    
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
        tools: List[AgentTool],
        context_variables: dict,
    ) -> Response:
        
        function_map = {f.__name__: f for f in tools}
        partial_response = Response(messages=[], agent=None, context_variables={})

        for tool_call in tool_calls:
            name = tool_call.function.name
            # handle missing tool case, skip to next tool
            if name not in function_map:
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

            func = function_map[name]
            # pass context_variables to agent functions
            if __CTX_VARS_NAME__ in func.__code__.co_varnames:
                args[__CTX_VARS_NAME__] = context_variables
            raw_result = function_map[name](**args)

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

    def invoke(
        self,
        messages: List,
        context_variables: dict,
        stream: bool = False,
    ):
        
        context_variables = defaultdict(str, context_variables)
        instructions = (
            self.instructions(context_variables)
            if callable(self.instructions)
            else self.instructions
        )

        messages = [{"role": "system", "content": instructions}] + messages

        tools = [function_to_json(f) for f in self.tools]
        # hide context_variables from model
        for tool in tools:
            params = tool["function"]["parameters"]
            params["properties"].pop(__CTX_VARS_NAME__, None)
            if __CTX_VARS_NAME__ in params["required"]:
                params["required"].remove(__CTX_VARS_NAME__)

        params = {
            "messages": messages,
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
    