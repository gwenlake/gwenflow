import os

from typing import Iterator, List

from gwenflow.logger import logger
from gwenflow.skills.base import Skill


class SkillsDirectory:
    """Loads and exposes all skills found in a directory of skill sub-directories.

    Each sub-directory that contains a ``SKILL.md`` file is treated as one
    skill. Sub-directories without a ``SKILL.md`` are silently skipped.

    Expected layout::

        <path>/
        ├── skill_a/
        │   ├── SKILL.md
        │   ├── scripts/
        │   └── resources/
        ├── skill_b/
        │   └── SKILL.md
        └── not_a_skill/      # ignored — no SKILL.md

    Example::

        sd = SkillsDirectory("./skills/")
        agent = Agent(
            name="MyAgent",
            skills=sd.skills,
            tools=sd.get_tools(),
        )
    """

    def __init__(self, path: str) -> None:
        if not os.path.isdir(path):
            raise NotADirectoryError(f"'{path}' is not a directory")
        self.path = path
        self.skills: List[Skill] = self._load()

    def _load(self) -> List[Skill]:
        skills: List[Skill] = []
        for entry in sorted(os.listdir(self.path)):
            entry_path = os.path.join(self.path, entry)
            if not os.path.isdir(entry_path):
                continue
            if not os.path.isfile(os.path.join(entry_path, "SKILL.md")):
                continue
            try:
                skills.append(Skill.from_directory(entry_path))
            except Exception as e:
                logger.warning(f"Failed to load skill from '{entry_path}': {e}")
        return skills

    def get_tools(self) -> list:
        """Return all script-defined tools from all skills in this directory."""
        return [tool for skill in self.skills for tool in skill.get_tools()]

    def to_toolset(self) -> "SkillsToolset":
        """Return a SkillsToolset wrapping all skills in this directory."""
        from gwenflow.skills.toolset import SkillsToolset
        return SkillsToolset(self.skills)

    def __iter__(self) -> Iterator[Skill]:
        return iter(self.skills)

    def __len__(self) -> int:
        return len(self.skills)

    def __repr__(self) -> str:
        return f"SkillsDirectory(path={self.path!r}, skills={[s.name for s in self.skills]})"
