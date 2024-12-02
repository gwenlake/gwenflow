
from typing import List
from pydantic import BaseModel

from gwenflow.agents import Agent
from gwenflow.utils import logger



class Flow(BaseModel):

    agents: List[Agent] = []
    flow_type: str = "sequence"


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
    