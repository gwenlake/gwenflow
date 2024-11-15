
from typing import List, Callable, Union, Optional, Any, Dict
from pydantic import BaseModel
import logging

from gwenflow.tools import Tool
from gwenflow.tasks import Task


logger = logging.getLogger(__name__)



class Flow(BaseModel):

    instructions: str = "You are a helpful AI system that can run a complex list of tasks."
    # llm: Any
    # tools: List[Tool] = []
    tasks: List[Any] = []
    flow_type: str = "sequence"

    # TODO
    # def auto(self, task=""):
    #     prompt = f"Generate a list of tasks, step by step, to answer the following question: { task }"
    #     task_list = self.llm

    def run(self) -> str:

        context = None

        for task in self.tasks:

            print("")
            print("------------------------------------------")
            print(f"Running Agent { task.agent.role }")
            print("------------------------------------------")

            context = task.run(context=context)

            print(f"{ context }")
        
        return context
    