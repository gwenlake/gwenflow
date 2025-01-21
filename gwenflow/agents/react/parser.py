
import re
from json_repair import repair_json

from gwenflow.agents.react.types import ActionReasoningStep


FINAL_ANSWER = "Final Answer:"


def _extract_thought(text: str) -> str:
    regex = r"(.*?)(?:\n\nAction|\n\nFinal Answer)"
    thought_match = re.search(regex, text, re.DOTALL)
    if thought_match:
        return thought_match.group(1).strip()
    return ""

def _clean_action(text: str) -> str:
    """Clean action string by removing non-essential formatting characters."""
    return re.sub(r"^\s*\*+\s*|\s*\*+\s*$", "", text).strip()

def _safe_repair_json(tool_input: str) -> str:
    UNABLE_TO_REPAIR_JSON_RESULTS = ['""', "{}"]
    if tool_input.startswith("[") and tool_input.endswith("]"):
        return tool_input
    tool_input = tool_input.replace('"""', '"')
    result = repair_json(tool_input)
    if result in UNABLE_TO_REPAIR_JSON_RESULTS:
        return tool_input
    return str(result)
    
def parse_reasoning_step(text: str) -> ActionReasoningStep:

    thought = _extract_thought(text)
    includes_answer = "Final Answer:" in text
    regex = (
        r"Action\s*\d*\s*:[\s]*(.*?)[\s]*Action\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
    )
    action_match = re.search(regex, text, re.DOTALL)

    if action_match:

        if includes_answer:
            raise ValueError("Error while trying to perform Action and give a Final Answer at the same time!")

        action = action_match.group(1)
        action = _clean_action(action)
        action_input = action_match.group(2).strip()

        tool_input = action_input.strip(" ").strip('"')
        tool_input = _safe_repair_json(tool_input)

        return ActionReasoningStep(thought=thought, action=action, action_input=tool_input)

    elif includes_answer:
        response = text.split(FINAL_ANSWER)[-1].strip()
        return ActionReasoningStep(thought=thought, response=response, is_done=True)

    if not re.search(r"Action\s*\d*\s*:[\s]*(.*?)", text, re.DOTALL):
        raise ValueError("Missing Action after Thought!")

    elif not re.search(r"[\s]*Action\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)", text, re.DOTALL):
        raise ValueError("Missing Action Input after Action!")
    else:
        raise ValueError("Sorry, I didn't use the right tool format.!")

