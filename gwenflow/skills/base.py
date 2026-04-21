import importlib.util
import inspect
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

import yaml

if TYPE_CHECKING:
    from gwenflow.tools import BaseTool


SKILL_TEMPLATE = """<skill>
<name>{name}</name>
<description>{description}</description>
<uri>{uri}</uri>

<resources>
{resources_list}
</resources>

<scripts>
{scripts_list}
</scripts>

<instructions>
{content}
</instructions>
</skill>
"""


@dataclass
class Skill:
    name: str
    description: str
    content: str = ""
    uri: Optional[str] = None
    license: Optional[str] = None
    compatibility: Optional[str] = None
    allowed_tools: Optional[str] = None
    resources: List[str] = field(default_factory=list)
    scripts: List[str] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        result = {"name": self.name, "description": self.description}
        if self.uri is not None:
            result["uri"] = self.uri
        if self.license is not None:
            result["license"] = self.license
        if self.compatibility is not None:
            result["compatibility"] = self.compatibility
        if self.allowed_tools is not None:
            result["allowed-tools"] = self.allowed_tools
        if self.content:
            result["content"] = self.content
        if self.resources:
            result["resources"] = self.resources
        if self.scripts:
            result["scripts"] = self.scripts
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    def to_prompt(self) -> str:
        return SKILL_TEMPLATE.format(
            name=self.name,
            description=self.description,
            uri=self.uri or "",
            resources_list="\n".join(self.resources),
            scripts_list="\n".join(self.scripts),
            content=self.content,
        )

    def get_tools(self) -> List["BaseTool"]:
        """Dynamically load tools from this skill's script files.

        Each script may expose tools in one of two ways:
        - A module-level ``tools`` list of ``BaseTool`` instances (preferred).
        - Any module-level function that has a docstring and type annotations
          (auto-wrapped as ``FunctionTool``).
        """
        from gwenflow.tools import BaseTool, FunctionTool

        all_tools: List[BaseTool] = []

        for script_path in self.scripts:
            spec = importlib.util.spec_from_file_location("_skill_script", script_path)
            if spec is None:
                continue
            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)
            except Exception as e:
                from gwenflow.logger import logger

                logger.warning(f"Failed to load skill script '{script_path}': {e}")
                continue

            # Prefer an explicit module-level 'tools' list
            if hasattr(module, "tools") and isinstance(module.tools, list):
                all_tools.extend(module.tools)
                continue

            # Fall back: auto-wrap public functions with docstrings + annotations
            for fn_name, fn in inspect.getmembers(module, inspect.isfunction):
                if fn_name.startswith("_"):
                    continue
                if fn.__doc__ and fn.__annotations__:
                    try:
                        all_tools.append(FunctionTool.from_function(fn))
                    except Exception:
                        pass

        return all_tools

    @staticmethod
    def _parse_skill_md(raw: str) -> tuple[dict, str]:
        """Split YAML frontmatter from markdown body in a SKILL.md file."""
        if raw.startswith("---"):
            parts = raw.split("---", 2)
            if len(parts) >= 3:
                frontmatter = yaml.safe_load(parts[1]) or {}
                body = parts[2].strip()
                return frontmatter, body
        return {}, raw.strip()

    @classmethod
    def from_dict(cls, data: dict) -> "Skill":
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            content=data.get("content", ""),
            uri=data.get("uri"),
            license=data.get("license"),
            compatibility=data.get("compatibility"),
            allowed_tools=data.get("allowed-tools") or data.get("allowed_tools"),
            resources=data.get("resources", []),
            scripts=data.get("scripts", []),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_yaml(cls, file: str) -> "Skill":
        """Load a skill from a standalone YAML file."""
        with open(file) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_directory(cls, path: str) -> "Skill":
        """Load a skill from a directory.

        Expected layout::

            <path>/
            ├── SKILL.md        # YAML frontmatter + markdown instructions
            ├── scripts/        # optional Python files that define tools
            │   └── *.py
            └── resources/      # optional resource files
                └── *

        ``SKILL.md`` format::

            ---
            name: MySkill
            description: What this skill does
            uri: https://example.com          # optional
            license: MIT                      # optional
            compatibility: ">=0.8.0"          # optional
            allowed-tools: ToolA,ToolB        # optional
            ---

            The skill instructions go here as plain markdown.
        """
        skill_md = os.path.join(path, "SKILL.md")
        if not os.path.isfile(skill_md):
            raise FileNotFoundError(f"SKILL.md not found in '{path}'")

        with open(skill_md) as f:
            frontmatter, content = cls._parse_skill_md(f.read())

        # Collect script paths
        scripts: List[str] = []
        scripts_dir = os.path.join(path, "scripts")
        if os.path.isdir(scripts_dir):
            for fname in sorted(os.listdir(scripts_dir)):
                if fname.endswith(".py") and not fname.startswith("_"):
                    scripts.append(os.path.join(scripts_dir, fname))

        # Collect resource paths
        resources: List[str] = []
        resources_dir = os.path.join(path, "resources")
        if os.path.isdir(resources_dir):
            for fname in sorted(os.listdir(resources_dir)):
                fpath = os.path.join(resources_dir, fname)
                if os.path.isfile(fpath):
                    resources.append(fpath)

        data = {**frontmatter, "content": content, "scripts": scripts, "resources": resources}
        return cls.from_dict(data)
