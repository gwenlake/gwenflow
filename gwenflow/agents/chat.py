
import uuid
import json
import inspect
import asyncio

from typing import List, Union, Optional, Any, Dict, Iterator
from collections import defaultdict
from pydantic import BaseModel, model_validator, field_validator, Field, ConfigDict, UUID4
from datetime import datetime

from gwenflow.logger import logger
from gwenflow.llms import ChatBase, ChatOpenAI
from gwenflow.types import Message
from gwenflow.agents import Agent



PROMPT_REFORMULATE = """\
# Instructions
- You are an AI assistant reading a current user query and chat_history.
- Given the chat_history, and current user's query, infer the user's intent expressed in the current user query.
- Once you infer the intent, respond with a task that can be used to with an Agent for the current user's query based on the intent
- Be specific in what the user is asking about, but disregard parts of the chat history that are not relevant to the user's intent.
- Provide responses in JSON format

# Examples
Example 1:
With a conversation like below:
```
 - user: are the trailwalker shoes waterproof?
 - assistant: Yes, the TrailWalker Hiking Shoes are waterproof. They are designed with a durable and waterproof construction to withstand various terrains and weather conditions.
 - user: how much do they cost?
```
Respond with:
{
    "intent": "The user wants to know how much the Trailwalker Hiking Shoes cost.",
    "task": "price of Trailwalker Hiking Shoes"
}

Example 2:
With a conversation like below:
```
 - user: are the trailwalker shoes waterproof?
 - assistant: Yes, the TrailWalker Hiking Shoes are waterproof. They are designed with a durable and waterproof construction to withstand various terrains and weather conditions.
 - user: how much do they cost?
 - assistant: The TrailWalker Hiking Shoes are priced at $110.
 - user: do you have waterproof tents?
 - assistant: Yes, we have waterproof tents available. Can you please provide more information about the type or size of tent you are looking for?
 - user: which is your most waterproof tent?
 - assistant: Our most waterproof tent is the Alpine Explorer Tent. It is designed with a waterproof material and has a rainfly with a waterproof rating of 3000mm. This tent provides reliable protection against rain and moisture.
 - user: how much does it cost?
```
Respond with:
{
    "intent": "The user would like to know how much the Alpine Explorer Tent costs.",
    "task": "price of Alpine Explorer Tent"
}

Return the task for the messages in the following conversation:

"""

class ChatAgent(BaseModel):

    llm: Optional[ChatBase] = Field(None, validate_default=True)
    agent: Agent

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    @field_validator("llm", mode="before")
    @classmethod
    def set_llm(cls, v: Optional[Any]) -> Any:
        llm = v or ChatOpenAI(model="gpt-4o-mini")
        return llm
    
    def run(self, messages: Union[str, List[Message], List[Dict[str, str]]]) -> Any:

        messages = self.llm._cast_messages(messages)

        conversation = []
        for message in messages:
            conversation.append(f"- {message.role}: {message.content}")
        conversation = "\n".join(conversation)
        prompt = PROMPT_REFORMULATE + f"<conversation>\n{conversation}\n</conversation>"

        self.llm.response_format = {"type": "json_object"}
        response = self.llm.invoke(prompt)
        response = response.choices[0].message.content

        logger.debug("Task: " + response["task"])

        return self.agent.run(response["task"])
