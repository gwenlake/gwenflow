import json
import re
from typing import Union, Optional, Any, Dict, List, Iterator
from collections import defaultdict

from gwenflow.types import ChatCompletionMessage
from gwenflow.agents.types import AgentResponse
from gwenflow.agents.agent import Agent
from gwenflow.agents.react.types import ActionReasoningStep
from gwenflow.agents.react.parser import ReActOutputParser
from gwenflow.agents.react.prompts import PROMPT_REACT
from gwenflow.agents.prompts import PROMPT_TASK, PROMPT_TOOLS
from gwenflow.utils.chunks import merge_chunk
from gwenflow.utils import logger


MAX_TURNS = float('inf')


class ReActAgent(Agent):

    is_react: bool = True
    description: str = "You are a meticulous and thoughtful assistant that solves a problem by thinking through it step-by-step."
    reasoning_steps: List[ActionReasoningStep] = []

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
            # logger.warning(f"Unknown tool {reasoning_step.action}, should be instead one of { tool_map.keys() }.")
            return {
                "role": "user",
                "content": "Ok. Let’s proceed with the next step.",
            }

        observation = self.execute_tool_call(reasoning_step.action, reasoning_step.action_input)
                
        return {
            "role": "user",
            "content": f"Observation: {observation}",
        }
    
    def invoke(self, messages: list, stream: bool = False) ->  Union[Any, Iterator[Any]]:

        params = {
            "messages": messages,
        }

        response_format = None
        if self.response_model:
            response_format = {"type": "json_object"}

        if stream:
            return self.llm.stream(**params, response_format=response_format)
        
        return self.llm.invoke(**params, response_format=response_format)

    def reason(self, task: str):

        if self.reasoning_model is None:
            return None
        
        user_prompt = ""
        
        if self.tools:
            tools = self.get_tools_text_schema()
            user_prompt += PROMPT_TOOLS.format(tools=tools).strip() + "\n\n"

        user_prompt += PROMPT_TASK.format(task=task).strip()
        user_prompt += "\n\nPlease help me with some thoughts, steps and guidelines to answer accurately and precisely to this task."

        params = {
            "messages": [{"role": "user", "content": user_prompt}],
        }

        logger.debug("Reasoning.")
        response = self.reasoning_model.invoke(**params)

        # only keep text outside <think>
        reasoning_content = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        if not reasoning_content:
            return None
        
        reasoning_content = reasoning_content.strip()
        
        logger.debug("Thought: " + reasoning_content)

        return reasoning_content

    def _run(
        self,
        task: Optional[str] = None,
        *,
        context: Optional[Any] = None,
        stream: Optional[bool] = False,
    ) ->  Iterator[AgentResponse]:

        # add reasoning
        if self.reasoning_model:
            reasoning_message = self.reason(task=task)
            if reasoning_message:
                if not context:
                    context = {}
                if isinstance(context, str):
                    text = context
                    context = { "context": text }
                context["thinking"] = reasoning_message # modifier en guidelines ?

        # messages for model
        user_message = self.get_user_message(task=task, context=context)
        self.memory.add_message(user_message)
        messages_for_model = self.get_messages_for_model(task=task, context=context)

        # system_message = self.get_system_message(context=context)
        # user_message = self.get_user_message(task=task, context=context)

        # check if system prompt is allow and add messages to messages_for_model
        # if self.system_prompt_allowed:
        #     if system_message:
        #         messages_for_model.append(system_message)
        #     if user_message:
        #         messages_for_model.append(user_message)
        #         self.memory.add_message(user_message)
        # else:
        #     system_message["role"] = "user"
        #     if user_message:
        #         system_message["content"] += "\n\n" + user_message["content"]
        #     messages_for_model.append(system_message)
        #     self.memory.add_message(system_message)
        
        # global loop
        init_len = len(messages_for_model)
        while len(messages_for_model) - init_len < MAX_TURNS:

            if stream:
                message = {
                    "content": "",
                    "role": "assistant",
                }

                completion = self.invoke(messages=messages_for_model, stream=True)

                for chunk in completion:
                    if len(chunk.choices) > 0:
                        delta = json.loads(chunk.choices[0].delta.json())
                        if delta["content"]:
                            yield AgentResponse(
                                delta=delta["content"],
                                messages=None,
                                agent=self,
                                tools=self.tools,
                            )
                        elif delta["tool_calls"] and self.show_tool_calls:
                            if delta["tool_calls"][0]["function"]["name"] and not delta["tool_calls"][0]["function"]["arguments"]:
                                response = f"""**Calling:** {delta["tool_calls"][0]["function"]["name"]}"""
                                yield AgentResponse(
                                    delta=response,
                                    messages=None,
                                    agent=self,
                                    tools=self.tools,
                                )
                        delta.pop("role", None)
                        merge_chunk(message, delta)

                message = ChatCompletionMessage(**message)
            
            else:
                completion = self.invoke(messages=messages_for_model)                
                message = completion.choices[0].message

            # add messages to the current message stack
            message_dict = json.loads(message.model_dump_json())
            messages_for_model.append(message_dict)

            # parse response
            reasoning_step = self.parse(message_dict["content"])
            self.reasoning_steps.append(reasoning_step)

            # show response
            logger.debug(f"{ reasoning_step.thought }")

            # final answer ?
            if reasoning_step.is_done:
                logger.debug("Task done.")
                self.memory.add_message(message_dict)
                break

            # handle tool calls
            observation = self.handle_tool_call(reasoning_step)
            if observation:
                messages_for_model.append(observation)

        content = messages_for_model[-1]["content"]
        if self.response_model:
            content = json.loads(content)

        if len(self.reasoning_steps)>0:
            last_reasoning_step = self.reasoning_steps[-1]
            content = last_reasoning_step.thought + "\n\n" + last_reasoning_step.response

        yield AgentResponse(
            content=content,
            messages=messages_for_model[init_len:],
            agent=self,
            tools=self.tools,
        )

    def get_reasoning_steps(self) -> str:
        steps = []
        steps.append("# Reasoning Steps")
        for i, reasoning_step in enumerate(self.reasoning_steps):
            if reasoning_step.is_done:
                text  = f"## Final Step\n"
                text += f"- **Reasoning:** {reasoning_step.thought}\n"
                text += f"- **Final Answer:** {reasoning_step.response}\n"
                steps.append(text)
                break
            else:
                text  = f"## Step {i+1}.\n"
                text += f"- **Reasoning:** {reasoning_step.thought}\n"
                text += f"- **Action:** {reasoning_step.action}, **Inputs:** {reasoning_step.action_input}\n"
                steps.append(text)

        steps = "\n\n".join(steps)
        return steps

