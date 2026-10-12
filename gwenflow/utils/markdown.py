"""Markdown helpers shared by skills (SKILL.md) and long-term memory files."""

from typing import Any

import yaml


def parse_frontmatter(raw: str) -> tuple[dict[str, Any], str]:
    """Split a YAML frontmatter block from the markdown body.

    A frontmatter block is a leading ``---`` line, YAML, and a closing ``---``
    line. Returns ``({}, raw.strip())`` when there is none, or when the YAML
    is not a mapping (a stray ``---`` horizontal rule must not eat the file).
    """
    if raw.startswith("---"):
        parts = raw.split("---", 2)
        if len(parts) >= 3:
            try:
                frontmatter = yaml.safe_load(parts[1]) or {}
            except yaml.YAMLError:
                frontmatter = None
            if isinstance(frontmatter, dict):
                return frontmatter, parts[2].strip()
    return {}, raw.strip()


def render_frontmatter(frontmatter: dict[str, Any], body: str) -> str:
    """Render ``frontmatter`` + ``body`` back into one markdown document."""
    if not frontmatter:
        return body.strip() + "\n"
    header = yaml.safe_dump(frontmatter, allow_unicode=True, sort_keys=False).strip()
    return f"---\n{header}\n---\n\n{body.strip()}\n"
