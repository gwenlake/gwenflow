
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, model_validator, field_validator, Field

import yaml

from gwenflow.logger import logger
from gwenflow.agents import Agent
from gwenflow.llms import ChatBase
from gwenflow.tools import BaseTool


MAX_TRIALS=5

class FlowStep(BaseModel):
    agent: Agent
    task: str | None = None
    depends_on: List[str] = []
    final_step: bool = False

class Flow(BaseModel):
    steps: Optional[list] = []
    manager: Optional[Agent] = None
    llm: ChatBase
    tools: List[BaseTool] = []

    flow_type: str = "sequence"

    @model_validator(mode="after")
    def model_valid(self) -> Any:
        if self.manager is None:
            self.manager = Agent(
                name="Team Manager",
                instructions= [
                    "You are the leader of a team of AI Agents.",
                    "Manage the team to complete the task in the best way possible.",
                    "Even though you don't perform tasks by yourself, you have a lot of experience in the field, which allows you to properly evaluate the work of your team members.",
                    "You must always validate the output of the other Agents and you can re-assign the task if you are not satisfied with the result.",
                ],
                tools=self.tools,
                llm=self.llm,
            )
        return self
        
    @classmethod
    def from_yaml(cls, file: str, tools: List[BaseTool], llm: Optional[Any] = None) -> "Flow":
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

                        depends_on = None
                        if _values.get("depends_on"):
                            depends_on = _values.get("depends_on")                            

                        agent = Agent(
                            name=name,
                            description=_values.get("description"),
                            response_model=_values.get("response_model"),
                            tools=_tools,
                            depends_on=depends_on,
                            llm=llm,
                        )
                        agents.append(agent)
                    return Flow(agents=agents)
                except Exception as e:
                    logger.error(repr(e))
        raise NotImplementedError(f"from_yaml not implemented for {cls.__name__}")
    
    def describe(self):
        for step in self.steps:
            step = FlowStep(**step)
            print("---")
            print(f"Agent  : {step.agent.name}")
            if step.depends_on:
                print(f"Depends on:", ",".join(step.depends_on))
            if step.agent.tools:
                available_tools = [ tool.name for tool in step.agent.tools ]
                print(f"Tools  :", ",".join(available_tools))


    def run(self, query: str) -> str:

        outputs = {}

        while len(outputs) < len(self.steps):

            for step in self.steps:

                step = FlowStep(**step)

                # check if already run
                if step.agent.name in outputs.keys():
                    continue

                # check agent dependancies
                if step.depends_on:
                    context = None
                    if any(outputs.get(agent_name) is None for agent_name in step.depends_on):
                        continue
                    if step.depends_on:
                        context = { f"{agent_name}": outputs[agent_name].content for agent_name in step.depends_on }            
                    if step.task:
                        outputs[step.agent.name] = step.agent.run(step.task, context=context)
                    else:
                        outputs[step.agent.name] = step.agent.run(query, context=context)
                
                # no dependency
                else:
                    if step.task:
                        outputs[step.agent.name] = step.agent.run(step.task)
                    else:
                        outputs[step.agent.name] = step.agent.run(query)

        return outputs
    