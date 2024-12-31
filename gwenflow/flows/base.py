
from typing import List, Dict, Any
from pydantic import BaseModel
import yaml

from gwenflow.agents import Agent
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

                        context_vars = None
                        if _values.get("context"):
                            context_vars = _values.get("context")

                        agent = Agent(
                            name=name,
                            role=_values.get("role"),
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
            print(f"Role   : {agent.role}")
            if agent.context_vars:
                print(f"Context:", ",".join(agent.context_vars))
            if agent.tools:
                available_tools = [ tool.name for tool in agent.tools ]
                print(f"Tools  :", ",".join(available_tools))

    def run(self, message: str) -> str:

        context_vars = {}

        first_agent = True

        for agent in self.agents:

            # if agent.context_vars:
            #     for context_var in agent.context_vars:
            #         if 
            
            # else:

            if first_agent:
                response = agent.run(message, context=context)
                first_agent = False

            else:
                response = agent.run(context=context)

            if response.content:
                context = response.content

            logger.debug(f"{ context }")
        
        return context
    