from typing import Tuple

import re
import dirtyjson as json

from gwenflow.agents.react.types import ActionReasoningStep
from gwenflow.utils import extract_json_str


def extract_tool_use(input_text: str) -> Tuple[str, str, str]:
    pattern = (
        r"\s*Thought: (.*?)\n+Action: ([^\n\(\) ]+).*?\n+Action Input: .*?(\{.*\})"
    )

    match = re.search(pattern, input_text, re.DOTALL)
    if not match:
        raise ValueError(f"Could not extract tool use from input text: {input_text}")

    thought = match.group(1).strip()
    action = match.group(2).strip()
    action_input = match.group(3).strip()
    return thought, action, action_input

def action_input_parser(json_str: str) -> dict:
    processed_string = re.sub(r"(?<!\w)\'|\'(?!\w)", '"', json_str)
    pattern = r'"(\w+)":\s*"([^"]*)"'
    matches = re.findall(pattern, processed_string)
    return dict(matches)

def extract_final_response(input_text: str) -> Tuple[str, str]:
    pattern = r"\s*Thought:(.*?)Final Answer:(.*?)(?:$)"

    match = re.search(pattern, input_text, re.DOTALL)
    if not match:
        raise ValueError(
            f"Could not extract final answer from input text: {input_text}"
        )

    thought = match.group(1).strip()
    answer = match.group(2).strip()
    return thought, answer

def parse_action_reasoning_step(output: str) -> ActionReasoningStep:
    # Weaker LLMs may generate ReActAgent steps whose Action Input are horrible JSON strings.
    # `dirtyjson` is more lenient than `json` in parsing JSON strings.
    thought, action, action_input = extract_tool_use(output)
    json_str = extract_json_str(action_input)
    # First we try json, if this fails we use ast
    try:
        action_input_dict = json.loads(json_str)
    except Exception:
        action_input_dict = action_input_parser(json_str)
    return ActionReasoningStep(
        thought=thought, action=action, action_input=action_input_dict
    )

class ReActOutputParser:
    """ReAct Output parser."""

    def parse(self, output: str) -> ActionReasoningStep:
        """Parse output from ReAct agent.

        We expect the output to be in one of the following formats:
        1. If the agent need to use a tool to answer the question:
            ```
            Thought: <thought>
            Action: <action>
            Action Input: <action_input>
            ```
        2. If the agent can answer the question without any tools:
            ```
            Thought: <thought>
            Final Answer: <answer>
            ```
        """

        # Agent directly outputs the answer instead of following the thought-answer format
        if "Thought:" not in output:
            return ActionReasoningStep(
                thought="(Implicit) I can answer without any more tools!",
                response=output,
            )

        # An "Action" should take priority over an "Answer"
        if "Action:" in output:
            return parse_action_reasoning_step(output)

        if "Final Answer:" in output:
            thought, answer = extract_final_response(output)
            return ActionReasoningStep(thought=thought, response=answer, is_done=True)

        raise ValueError(f"Could not parse output: {output}")

