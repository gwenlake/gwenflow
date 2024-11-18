
from typing import List, Callable, Union, Optional, Any, Dict
from pydantic import BaseModel
import logging

from gwenflow.tools import Tool
from gwenflow.tasks import Task


logger = logging.getLogger(__name__)



class Flow(BaseModel):

    instructions: str = "You are a helpful AI system that can run a complex list of tasks."
    llm: Any = None
    tools: List[Tool] = []
    tasks: List[Any] = []
    flow_type: str = "sequence"


    def run(self) -> str:

        context = None

        for task in self.tasks:

            tools = [ tool.name for tool in task.agent.tools ]
            tools = ",".join(tools)

            print("")
            print("------------------------------------------")
            print(f"Task : { task.description }")
            print(f"Agent: { task.agent.role }")
            print(f"Tools: { tools }")
            print("------------------------------------------")

            context = task.run(context=context)

            print(f"{ context }")
        
        return context
    