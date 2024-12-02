
from typing import List
from pydantic import BaseModel
import yaml

from gwenflow.agents import Agent
from gwenflow.utils import logger



class Flow(BaseModel):

    agents: List[Agent] = []
    flow_type: str = "sequence"


    @classmethod
    def from_yaml(cls, file: str) -> "Flow":
        if cls == Flow:
            with open(file) as stream:
                try:
                    agents = []
                    content_yaml = yaml.safe_load(stream)
                    for name in content_yaml.get("agents").keys():
                        _values = content_yaml["agents"][name]
                        agent = Agent(
                            name=name,
                            role=_values.get("role"),
                            task=_values.get("task"),
                            response_model=_values.get("response_model"),
                            # tools=[tool_wikipedia],
                        )
                        agents.append(agent)
                    return Flow(agents=agents)
                except Exception as e:
                    logger.error(repr(e))

        raise NotImplementedError(f"from_yaml not implemented for {cls.__name__}")
    
    def run(self, message: str) -> str:

        context = None

        first_agent = True

        for agent in self.agents:

            logger.debug("")
            logger.debug("------------------------------------------")
            logger.debug(f"Agent: { agent.role }")
            logger.debug("------------------------------------------")

            if first_agent:
                response = agent.run(message, context=context)
                first_agent = False

            else:
                response = agent.run(context=context)

            if response.content:
                context = response.content

            logger.debug(f"{ context }")
        
        return context
    