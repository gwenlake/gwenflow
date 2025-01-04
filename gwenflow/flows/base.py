
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, model_validator, field_validator, Field

import yaml
import time

from gwenflow.agents import Agent
from gwenflow.agents.run import RunResponse
from gwenflow.tools import Tool
from gwenflow.utils import logger


MAX_TRIALS=5


class Flow(BaseModel):

    agents: List[Agent] = []
    manager: Optional[Agent] = Field(None, validate_default=True)
    flow_type: str = "sequence"

    @field_validator("manager", mode="before")
    def set_manager(cls, v: Optional[str]) -> str:
        manager = Agent(
            name="Team Manager",
            task="Manage the team to complete the task in the best way possible.",
            instructions= [
                "You are the leader of a team of AI Agents.",
                "Even though you don't perform tasks by yourself, you have a lot of experience in the field, which allows you to properly evaluate the work of your team members.",
                "You must always validate the output of the other Agents and you can re-assign the task if you are not satisfied with the result.",
            ]
        )
        return manager
    
    @classmethod
    def from_yaml(cls, file: str, tools: List[Tool]) -> "Flow":
        if cls == Flow:
            with open(file) as stream:
                try:
                    agents = []
                    content_yaml = yaml.safe_load(stream)
                    for name in content_yaml.get("agents").keys():

                        _values = content_yaml["agents"][name]

                        _tools = []
                        if _values.get("tools"):
                            _agent_tools = _values.get("tools").split(",")
                            for t in tools:
                                if t.name in _agent_tools:
                                    _tools.append(t)

                        context_vars = []
                        if _values.get("context"):
                            context_vars = _values.get("context")

                        agent = Agent(
                            name=name,
                            task=_values.get("task"),
                            response_model=_values.get("response_model"),
                            tools=_tools,
                            context_vars=context_vars,
                        )
                        agents.append(agent)
                    return Flow(agents=agents)
                except Exception as e:
                    logger.error(repr(e))
        raise NotImplementedError(f"from_yaml not implemented for {cls.__name__}")
    
    def describe(self):
        for agent in self.agents:
            print("---")
            print(f"Agent  : {agent.name}")
            if agent.task:
                print(f"Task   : {agent.task}")
            if agent.context_vars:
                print(f"Context:", ",".join(agent.context_vars))
            if agent.tools:
                available_tools = [ tool.name for tool in agent.tools ]
                print(f"Tools  :", ",".join(available_tools))


    def run(self, query: str) -> str:

        outputs = {}

        while len(outputs) < len(self.agents):

            for agent in self.agents:

                # check if already run
                if agent.name in outputs.keys():
                    continue

                # check agent dependancies
                if any(outputs.get(var) is None for var in agent.context_vars):
                    continue

                # validation by the manager
                trials = 1
                last_response = None
                while trials < MAX_TRIALS:

                    # prepare context and run
                    context = None
                    if agent.context_vars:
                        context = { f"{var}": outputs[var].content for var in agent.context_vars }
                    
                    task = None
                    if context is None:
                        task = query # always keep query if no context (first agents)

                    if last_response:
                        task = f"Task: {task}\n\nYour previous resonse this query was the following: {last_response}\n\n. Please, improve substantially the quality of this answer."

                    agent_response = agent.run(task=task, context=context)

                    manager_prompt = f"The task of the agent is: { agent.task }\n\nIf you validate the agent response simply answer 'Yes'."
                    manager_response = self.manager.run(manager_prompt, context=agent_response.content)
                    if manager_response.content == "Yes":
                        outputs[agent.name] = agent_response
                        break
                    
                    last_response = agent_response.content
                    logger.debug(f"{agent.name}: {last_response}")

                    trials += 1
                

        return outputs
    