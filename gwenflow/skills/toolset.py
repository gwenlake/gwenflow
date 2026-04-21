import os
from typing import TYPE_CHECKING, List

from gwenflow.skills.base import Skill

if TYPE_CHECKING:
    from gwenflow.tools import BaseTool


_INSTRUCTIONS_HEADER = """\
You have access to a collection of skills containing domain-specific knowledge and capabilities.

<available_skills>
{skills_list}
</available_skills>

When a task falls within a skill's domain:
1. Use `load_skill` first to read the complete instructions for that skill.
2. Only after `load_skill` succeeds, follow the skill's guidance to complete the task.
3. Call `read_skill_resource` only for skills already loaded with `load_skill`.
4. Never guess resource names; use the exact names listed by `load_skill`.\
"""


class SkillsToolset:
    """Manages a set of skills and exposes them as agent tools.

    Provides three tools for the agent:

    - ``list_skills``: lists all available skills with their descriptions.
    - ``load_skill``: loads full instructions for a specific skill on demand.
    - ``read_skill_resource``: reads a resource file from a specific skill.

    The agent receives a compact listing in the system prompt and loads skill
    details only when needed (progressive disclosure), keeping token usage low.

    Example::

        toolset = SkillsToolset(skills)
        agent = Agent(
            name="MyAgent",
            skills=skills,  # compact listing + management tools added automatically
            tools=toolset.get_tools(),  # or pass explicitly
        )
    """

    def __init__(self, skills: List[Skill]) -> None:
        self._skills: dict[str, Skill] = {s.name: s for s in skills}

    def get_instructions(self) -> str:
        """Return a compact system prompt section listing available skills by name and description."""
        if not self._skills:
            return ""
        lines: List[str] = []
        for skill in self._skills.values():
            lines.append("<skill>")
            lines.append(f"<name>{skill.name}</name>")
            lines.append(f"<description>{skill.description}</description>")
            if skill.uri:
                lines.append(f"<uri>{skill.uri}</uri>")
            lines.append("</skill>")
        return _INSTRUCTIONS_HEADER.format(skills_list="\n".join(lines))

    def get_tools(self) -> List["BaseTool"]:
        """Return the three skill management tools to register on the agent."""
        from gwenflow.tools import FunctionTool

        skills = self._skills

        def list_skills() -> str:
            """List all available skills with their name and description."""
            if not skills:
                return "No skills available."
            return "\n".join(f"- {name}: {s.description}" for name, s in skills.items())

        def load_skill(skill_name: str) -> str:
            """Load the full instructions for a named skill. Always call this before applying a skill."""
            skill = skills.get(skill_name)
            if skill is None:
                available = ", ".join(sorted(skills.keys()))
                return f"Skill '{skill_name}' not found. Available skills: {available}"
            return skill.to_prompt()

        def read_skill_resource(skill_name: str, resource_name: str) -> str:
            """Read a resource file from a skill. Call load_skill first to get exact resource names."""
            skill = skills.get(skill_name)
            if skill is None:
                return f"Skill '{skill_name}' not found."
            for path in skill.resources:
                if os.path.basename(path) == resource_name:
                    try:
                        with open(path) as f:
                            return f.read()
                    except Exception as e:
                        return f"Failed to read resource '{resource_name}': {e}"
            available = [os.path.basename(p) for p in skill.resources]
            return f"Resource '{resource_name}' not found in skill '{skill_name}'. Available: {available}"

        return [
            FunctionTool.from_function(list_skills),
            FunctionTool.from_function(load_skill),
            FunctionTool.from_function(read_skill_resource),
        ]
