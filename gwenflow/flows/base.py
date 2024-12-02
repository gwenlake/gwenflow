
from typing import List
from pydantic import BaseModel

from gwenflow.agents import Agent
from gwenflow.utils import logger



class Flow(BaseModel):

    agents: List[Agent] = []
    flow_type: str = "sequence"


    def run(self) -> str:

        context = None

        for agent in self.team_agents:

            logger.debug("")
            logger.debug("------------------------------------------")
            logger.debug(f"Agent: { agent.agents }")
            logger.debug("------------------------------------------")

            context = agent.run(message=agent.task, context=context)

            logger.debug(f"{ context }")
        
        return context
    