
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

import yaml
import time

from gwenflow.agents import Agent
from gwenflow.agents.run import RunResponse
from gwenflow.tools import Tool
from gwenflow.utils import logger


class Flow(BaseModel):

    agents: List[Agent] = []
    flow_type: str = "sequence"


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

                # prepare context and run
                context = None
                if agent.context_vars:
                    context = { f"{var}": outputs[var].content for var in agent.context_vars }
                outputs[agent.name] = agent.run(user_prompt=query, context=context)

        return outputs
    