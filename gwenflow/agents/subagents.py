from typing import List

SPAWN_AGENT_TOOL = "spawn_agent"

SUBAGENT_NAME = "sub-agent"
SUBAGENT_NAME_MAX_CHARS = 60

SUBAGENT_EVENT_POLL_SECONDS = 0.05

SUBAGENT_ABANDONED = (
    "[sub-agent failed] The sub-agent was waiting for the user's approval, "
    "and the user sent a new message instead: it was stopped."
)

FAST_MODEL = "fast"
STRONG_MODEL = "strong"
SUBAGENT_MODELS = (FAST_MODEL, STRONG_MODEL)

SUBAGENT_INSTRUCTIONS = [
    "You are a sub-agent launched by an orchestrator agent to complete one self-contained task.",
    "Work autonomously with your tools: nobody can answer a question from you during the task.",
    "Your final message is returned to the orchestrator, not to the user. Make it a concise report "
    "with what the orchestrator needs (findings, figures, sources, file paths, conclusions), "
    "not the raw data you went through.",
    "If you cannot complete the task, say what is missing and what you found so far.",
]

SPAWN_AGENT_DESCRIPTION = """\
Launch a sub-agent that completes a self-contained task on its own and returns its final report.

The sub-agent starts with an empty context: it does not see this conversation. The prompt must \
hold everything it needs: the goal, the inputs, the constraints and the expected shape of the \
report. Only its final report comes back to you, so use it for tasks that need many tool calls \
whose intermediate results you do not need to keep (searches, reading many documents, \
independent sub-questions).

Give it, by name, only the tools the task needs: it gets no other tool, and the fewer tools it \
has, the cheaper and more focused it is.{skills_hint}{model_hint}

To get the report as JSON with a fixed shape (figures to compare, lists to merge), pass an \
output_schema: a JSON Schema the final report must follow.

To run several sub-agents in parallel, call this tool several times in the same turn. Do not \
use it for a task you can do yourself in one or two tool calls. Sub-agents cannot launch other \
sub-agents. When a sub-agent calls a tool that needs the user's approval, the run pauses until \
the user answers, then the sub-agent resumes."""

SKILLS_HINT = (
    " Hand it the skills the task calls for the same way, by name: it can then load their "
    "instructions and use their scripts."
)

MODEL_HINT = (
    f" Pick the model too: '{FAST_MODEL}' (default) is cheaper and quicker, enough for searches, "
    f"lookups and extraction; '{STRONG_MODEL}' is your own model, for tasks that need careful reasoning."
)


def spawn_agent_description(with_model: bool, with_skills: bool = False) -> str:
    return SPAWN_AGENT_DESCRIPTION.format(
        skills_hint=SKILLS_HINT if with_skills else "",
        model_hint=MODEL_HINT if with_model else "",
    )


def spawn_agent_parameters(tool_names: List[str], skill_names: List[str], with_model: bool) -> dict:
    properties: dict = {
        "description": {
            "type": "string",
            "description": "A short (3-5 words) description of the task.",
        },
        "prompt": {
            "type": "string",
            "description": "The complete, self-contained task for the sub-agent.",
        },
    }
    required = ["description", "prompt"]
    if tool_names:
        properties["tools"] = {
            "type": "array",
            "items": {"type": "string", "enum": tool_names},
            "description": "Names of the tools the sub-agent may use: only those the task needs, [] for none.",
        }
        required.append("tools")
    if skill_names:
        properties["skills"] = {
            "type": "array",
            "items": {"type": "string", "enum": skill_names},
            "description": "Names of the skills the sub-agent may load: only those the task needs.",
        }
    properties["output_schema"] = {
        "type": "object",
        "description": "Optional JSON Schema the final report must follow; the report then comes back as JSON.",
    }
    if with_model:
        properties["model"] = {
            "type": "string",
            "enum": list(SUBAGENT_MODELS),
            "description": f"The model of the sub-agent. Defaults to '{FAST_MODEL}'.",
        }
    return {"type": "object", "properties": properties, "required": required}
