import logging
import json
from typing import List, Callable, Union, Any

from gwenflow.agents.agent import Agent, RunResponse
from gwenflow.tasks.prompts import CONTEXT, EXPECTED_OUTPUT


MAX_LOOPS = 10


logger = logging.getLogger(__name__)


class Task:
    
    def __init__(self, *, description: str, expected_output: str = None, agent: Agent):

        self.id = None
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


    def run(self, context: str = None) -> str:
        
        task_prompt  = self.prompt(context)
        active_agent = self.agent
        
        num_loops = 1
        while active_agent and num_loops < MAX_LOOPS:

            response = active_agent.run(task_prompt, context=context)
            
            # task done
            if isinstance(response, RunResponse):
                response = response.content
                break

            # task transfered to another agent
            elif isinstance(response, Agent):
                active_agent = response

            num_loops += 1

        return response
