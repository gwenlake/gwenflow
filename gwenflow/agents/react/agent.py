import json
from typing import Union, Optional, Any, Dict, Iterator
from collections import defaultdict

from gwenflow.types import ChatCompletionMessage
from gwenflow.agents.types import AgentResponse
from gwenflow.agents.agent import Agent
from gwenflow.agents.react.types import ActionReasoningStep
from gwenflow.agents.react.parser import ReActOutputParser
from gwenflow.agents.react.prompts import PROMPT_REACT
from gwenflow.agents.utils import merge_chunk
from gwenflow.utils import logger


MAX_TURNS = float('inf')


class ReActAgent(Agent):

    is_react: bool = True
    description: str = "You are a meticulous and thoughtful assistant that solves a problem by thinking through it step-by-step."


    def parse(self, text: str) -> ActionReasoningStep:
        return ReActOutputParser().parse(text)

    def get_system_message(self, context: Optional[Any] = None):
        """Return the system message for the Agent."""

        # Add additional instructions
        additional_guidelines = [
            "Your goal is to reason about the task or query and decide on the best course of action to answer it accurately.",
            "If you cannot find the necessary information after using available tools, admit that you don't have enough information to answer the query confidently.",
        ]
        self.instructions = additional_guidelines + self.instructions

        # default system message
        system_message = super(ReActAgent, self).get_system_message(context=context)

        # ReAct
        tool_names = ",".join(self.get_tool_names())
        prompt = PROMPT_REACT.format(tool_names=tool_names).strip()
        system_message["content"] += f"\n\n{prompt}"

        return system_message

    def handle_tool_call(self, reasoning_step: ActionReasoningStep) -> Dict:
        
        tool_map = self.get_tools_map()

        # handle missing tool case, skip to next tool
        if reasoning_step.action not in tool_map:

            logger.warning(f"Unknown tool {reasoning_step.action}, should be instead one of { tool_map.keys() }.")

            if reasoning_step.action_input:
                return {
                    "role": "user",
                    "content": f"Observation: { json.dumps(reasoning_step.action_input) }.",
                }
            else:
                return {
                    "role": "user",
                    "content": "Observation: None. Let’s proceed to the next step.",
                }

        observation = self.execute_tool_call(reasoning_step.action, reasoning_step.action_input)
                
        return {
            "role": "user",
            "content": f"Observation: {observation}",
        }
    
    def invoke(self, messages: list, stream: bool = False) ->  Union[Any, Iterator[Any]]:

        params = {
            "messages": messages,
            "parse_response": False,
        }

        response_format = None
        if self.response_model:
            response_format = {"type": "json_object"}

        if stream:
            return self.llm.stream(**params, response_format=response_format)
        
        return self.llm.invoke(**params, response_format=response_format)


    def _run(
        self,
        task: Optional[str] = None,
        *,
        context: Optional[Any] = None,
        stream: Optional[bool] = False,
    ) ->  Iterator[AgentResponse]:

        messages_for_model = []

        # system messages
        system_message = self.get_system_message(context=context)
        if system_message:
            messages_for_model.append(system_message)

        # user messages
        user_message = self.get_user_message(task=task, context=context)
        if user_message:
            messages_for_model.append(user_message)
            self.memory.add_message(user_message)
       
        # global loop
        reasoning_step = None
        init_len = len(messages_for_model)
        while len(messages_for_model) - init_len < MAX_TURNS:

            if stream:
                message = {
                    "content": "",
                    "sender": self.name,
                    "role": "assistant",
                }

                completion = self.invoke(messages=messages_for_model, stream=True)

                for chunk in completion:
                    if len(chunk.choices) > 0:
                        delta = json.loads(chunk.choices[0].delta.json())
                        if delta["role"] == "assistant":
                            delta["sender"] = self.name
                        if delta["content"]:
                            yield AgentResponse(
                                delta=delta["content"],
                                messages=None,
                                agent=self,
                                tools=self.tools,
                            )
                        delta.pop("role", None)
                        delta.pop("sender", None)
                        merge_chunk(message, delta)

                message = ChatCompletionMessage(**message)
            
            else:
                completion = self.invoke(messages=messages_for_model)                
                message = completion.choices[0].message
                message.sender = self.name

            # add messages to the current message stack
            message_dict = json.loads(message.model_dump_json())
            messages_for_model.append(message_dict)

            # parse response
            reasoning_step = self.parse(message_dict["content"])

            # show response
            logger.debug(f"\n---\nThought: { reasoning_step.thought }")

            # final answer ?
            if reasoning_step.is_done:
                logger.debug("Task done.")
                self.memory.add_message(message_dict)
                break

            # handle tool calls
            observation = self.handle_tool_call(reasoning_step)
            messages_for_model.append(observation)

        content = messages_for_model[-1]["content"]
        if self.response_model:
            content = json.loads(content)

        if reasoning_step:
            content = reasoning_step.thought + "\n\n" + reasoning_step.response

        yield AgentResponse(
            content=content,
            messages=messages_for_model[init_len:],
            agent=self,
            tools=self.tools,
        )

