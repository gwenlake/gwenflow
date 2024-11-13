
from typing import List, Callable, Union, Optional, Any
from pydantic import BaseModel


AgentTool = Callable[[], Union[str, "Agent", dict]]


class Response(BaseModel):
    output: Any = None
    messages: List = []
    agent: Optional[Any] = None
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
    agent: Optional[Any] = None
    context_variables: dict = {}