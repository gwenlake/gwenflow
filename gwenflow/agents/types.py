
from typing import List, Dict, Callable, Union, Optional, Any
from pydantic import BaseModel, Field
from time import time


AgentTool = Callable[[], Union[str, "Agent", dict]]


class RunResponse(BaseModel):
    content: Optional[Any] = None
    content_type: str = "str"
    messages: Optional[List] = None
    agent: Optional[Any] = None
    tools: Optional[List[Any]] = None
    created_at: int = Field(default_factory=lambda: int(time()))


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