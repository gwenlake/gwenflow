
import uuid
import json
import asyncio
# import mem0

from typing import List, Union, Optional, Any, Dict, Iterator, Literal
from pydantic import BaseModel, model_validator, field_validator, Field, ConfigDict, UUID4
from datetime import datetime

from gwenflow.llms import ChatBase, ChatOpenAI, ModelResponse
from gwenflow.types import Message
from gwenflow.tools import BaseTool
from gwenflow.memory import ChatMemoryBuffer
from gwenflow.knowledge import Knowledge
from gwenflow.agents.types import AgentResponse
from gwenflow.agents.prompts import PROMPT_ROLE, PROMPT_STEPS, PROMPT_INSTRUCTIONS, PROMPT_JSON_SCHEMA, PROMPT_CONTEXT, PROMPT_KNOWLEDGE
from gwenflow.utils import logger
from gwenflow.types import Message


class Agent(BaseModel):

    id: UUID4 = Field(default_factory=uuid.uuid4, frozen=True)
    """The unique id of the agent."""

    name: str
    """The name of the agent."""

    description: str | None = None
    """A description of the agent, used as a handoff, so that the manager knows what it does."""

    instructions: (str | List[str] | None) = None
    """The instructions for the agent."""

    response_model: Dict | None = None
    """Response model."""

    llm: Optional[ChatBase] = Field(None, validate_default=True)
    """The model implementation to use when invoking the LLM."""

    tools: List[BaseTool] | None = None
    """A list of tools that the agent can use."""
    
    tool_choice: Literal["auto", "required", "none"] | str | None = None
    """The tool choice to use when calling the model."""

    reasoning_model: Optional[Any] = Field(None, validate_default=True)
    """Reasoning model."""

    reasoning_steps: str | None = None
    """Reasoning steps."""

    history: ChatMemoryBuffer | None = None
    """Historcal messages for the agent."""

    knowledge: Optional[Knowledge] = None
    """Knowledge for the agent."""

    team: List["Agent"] | None = None
    """Team of agents."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("id", mode="before")
    @classmethod
    def deny_user_set_id(cls, v: Optional[UUID4]) -> None:
        if v:
            raise ValueError("This field is not to be set by the user.")

    @field_validator("instructions", mode="before")
    @classmethod
    def set_instructions(cls, v: Optional[Union[List, str]]) -> str:
        if isinstance(v, str):
            instructions = [v]
            return instructions
        return v

    @field_validator("llm", mode="before")
    @classmethod
    def set_llm(cls, v: Optional[Any]) -> Any:
        llm = v or ChatOpenAI(model="gpt-4o-mini")
        return llm

    @model_validator(mode="after")
    def model_valid(self) -> Any:
        if self.llm:
            if self.history is None:
                token_limit = self.llm.get_context_window_size()
                self.history = ChatMemoryBuffer(token_limit=token_limit)
            if self.response_model:
                self.llm.response_format = {"type": "json_object"}
            if self.tools:
                self.llm.tools = self.tools
                self.llm.tool_choice = self.tool_choice
        # if self.short_term_memory is None:
        #      self.short_term_memory = mem0.Memory()
        return self

    def _format_context(self, context: Optional[Union[str, Dict[str, str]]]) -> str:
        text = ""
        if isinstance(context, str):
            text = f"<context>\n{ context }\n</context>\n\n"
        elif isinstance(context, dict):
            for key in context.keys():
                text += f"<{key}>\n"
                text += context.get(key) + "\n"
                text += f"</{key}>\n\n"
        return text
    
    def get_system_prompt(self, context: Optional[Union[str, Dict[str, str]]] = None, references: Optional[List[str]] = None) -> str:
        """Get the system prompt for the agent."""

        prompt = PROMPT_ROLE.format(name=self.name, date=datetime.now()).strip()
        prompt += "\n\n"

        # instructions
        instructions = []
        if self.response_model:
            instructions.append("Use JSON to format your answers.")    
        if context is not None:
            instructions.append("Always prefer information from the provided context over your own knowledge.")
        if self.instructions:
            if isinstance(self.instructions, str):
                instructions += [self.instructions]
            elif isinstance(self.instructions, list):
                instructions += self.instructions
        if len(instructions) > 0:
            instructions = "\n".join([f"- {i}" for i in instructions])
            prompt += PROMPT_INSTRUCTIONS.format(instructions=instructions).strip()
            prompt += "\n\n"

        # reasoning steps
        if self.reasoning_steps:
            prompt += PROMPT_STEPS.format(reasoning_steps=self.reasoning_steps).strip()
            prompt += "\n\n"

        # response model
        if self.response_model:
            prompt += PROMPT_JSON_SCHEMA.format(json_schema=json.dumps(self.response_model, indent=4)).strip()
            prompt += "\n\n"

        # references
        if references:
            prompt += PROMPT_KNOWLEDGE.format(references="\n\n".join(references)).strip()
            prompt += "\n\n"

        # context
        if context is not None:
            prompt += PROMPT_CONTEXT.format(context=self._format_context(context)).strip()
            prompt += "\n\n"

        return prompt.strip()
    
    # def reason(self, task: str):

    #     if self.reasoning_model is None:
    #         return None
        
    #     user_prompt = ""
        
    #     if self.tools:
    #         tools = self.get_tools_text_schema()
    #         user_prompt += PROMPT_TOOLS.format(tools=tools).strip() + "\n\n"

    #     user_prompt += PROMPT_TASK.format(task=task).strip()
    #     user_prompt += "\n\nPlease help me with some thoughts, steps and guidelines to answer accurately and precisely to this task."

    #     params = {
    #         "messages": [{"role": "user", "content": user_prompt}],
    #     }

    #     logger.debug("Reasoning.")
    #     response = self.reasoning_model.invoke(**params)

    #     # only keep text outside <think>
    #     reasoning_content = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    #     if not reasoning_content:
    #         return None
        
    #     reasoning_content = reasoning_content.strip()
        
    #     logger.debug("Thought: " + reasoning_content)

    #     return reasoning_content
    
    def _run(
        self,
        messages: Union[str, List[Message], List[Dict[str, str]]],
        context: Optional[Union[str, Dict[str, str]]] = None,
        stream: Optional[bool] = False,
    ) ->  Iterator[AgentResponse]:

        messages = self.llm._cast_messages(messages)

        # init agent and model response
        agent_response = AgentResponse()
        model_response = ModelResponse()

        # add reasoning steps
        # if self.reasoning_model:
        #     tools = self.get_tools_text_schema()
        #     completion = self.invoke(PROMPT_REASONING_STEPS_TOOLS.format(task=task, tools=tools))
        #     if len(completion.choices)>0:
        #         self.reasoning_steps = completion.choices[0].message.content

        # search for references
        references = None
        if self.knowledge:
            query = messages[-1].content
            references = self.knowledge.search(query=query, limit=20)
            if len(references)>0:
                references = [r.content for r in references]

        # history
        self.history.system_prompt = self.get_system_prompt(context=context, references=references)
        self.history.add_messages(messages)

        # format messages
        messages_for_model = [m.to_dict() for m in self.history.get()]

        # call llm and tool
        if stream:
            for chunk in self.llm.response_stream(messages=messages_for_model):
                agent_response.content = None
                agent_response.thinking = None
                if chunk.content:
                    model_response.content += chunk.content
                    agent_response.content  = chunk.content
                if chunk.thinking:
                    model_response.thinking += chunk.thinking
                    agent_response.thinking  = chunk.thinking
                yield agent_response
        
        else:
            model_response = self.llm.response(messages=messages_for_model)
            agent_response.content = model_response.content

        # add messages to the current message stack
        self.history.add_message(Message(role="assistant", content=model_response.content))

        # format response
        if self.response_model:
            agent_response.content = json.loads(agent_response.content)

        agent_response.finish_reason = "stop"

        yield agent_response


    def run(
        self,
        messages: Union[str, List[Message], List[Dict[str, str]]],
        context: Optional[Union[str, Dict[str, str]]] = None,
        stream: Optional[bool] = False,
    ) ->  Union[AgentResponse, Iterator[AgentResponse]]:

        logger.debug("")
        logger.debug("------------------------------------------")
        logger.debug(f"Running Agent: { self.name }")
        logger.debug("------------------------------------------")
        logger.debug("")

        if stream:
            response = self._run(messages=messages, context=context, stream=True)
            return response
    
        response = self._run(messages=messages, context=context, stream=False)
        return next(response)
