
from typing import List, Callable, Union, Optional, Any
from pydantic import BaseModel


AgentTool = Callable[[], Union[str, "Agent", dict]]


class Agent(BaseModel):
    role: str
    instructions: Union[str, Callable[[], str]] = "You are a helpful agent."
    llm: Any
    tools: List[AgentTool] = []
    tool_choice: str = None
    parallel_tool_calls: bool = True


class Response(BaseModel):
    output: Any = None
    messages: List = []
    agent: Optional[Agent] = None
    context_variables: dict = {}


class Result(BaseModel):
    """
    Encapsulates the possible return values for an agent function.

    Attributes:
        value (str): The result value as a string.
        agent (Agent): The agent instance, if applicable.
        context_variables (dict): A dictionary of context variables.
    """

    value: str = ""
    agent: Optional[Agent] = None
    context_variables: dict = {}