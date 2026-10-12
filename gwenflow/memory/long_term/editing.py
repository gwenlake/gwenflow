"""Text operations on memory files: edit, tidy, truncate. No I/O."""

from __future__ import annotations

import re

from gwenflow.utils.markdown import parse_frontmatter, render_frontmatter

_PLACEHOLDER_RE = re.compile(r"^\s*[-*] \(.*\)\s*$")


def line_count(content: str) -> int:
    return len(content.splitlines())


def truncate(content: str, max_chars: int) -> str:
    """Cut at a line boundary under ``max_chars`` and say how much was left out."""
    if len(content) <= max_chars:
        return content
    head = content[:max_chars]
    cut = head.rfind("\n")
    if cut > 0:
        head = head[:cut]
    dropped = line_count(content) - line_count(head)
    return f"{head}\n[truncated: {dropped} more lines, use read_memory for the full file]"


def _is_item(line: str) -> bool:
    return line.lstrip().startswith(("- ", "* "))


def _item_key(line: str) -> str:
    return " ".join(line.strip()[2:].strip().lower().rstrip(".;").split())


def tidy(content: str) -> str:
    """Clean a memory file after a write, section by section (sections start at headings).

    - A list item repeated in the same section is dropped (same text up to case,
      spacing and a final period).
    - A template placeholder, a list item fully in parentheses such as
      ``- (role, organisation)``, is dropped once its section holds a real item.
    """
    sections: list[list[str]] = [[]]
    for line in content.splitlines():
        if line.lstrip().startswith("#"):
            sections.append([line])
        else:
            sections[-1].append(line)
    out: list[str] = []
    for section in sections:
        has_real_item = any(_is_item(line) and not _PLACEHOLDER_RE.match(line) for line in section)
        seen: set[str] = set()
        for line in section:
            if _is_item(line):
                if has_real_item and _PLACEHOLDER_RE.match(line):
                    continue
                key = _item_key(line)
                if key in seen:
                    continue
                seen.add(key)
            out.append(line)
    result = "\n".join(out)
    return result + "\n" if content.endswith("\n") else result


def apply_edit(content: str, old_str: str, new_str: str) -> tuple[str | None, str | None]:
    """Replace the single occurrence of ``old_str``. Returns ``(new_content, error)``."""
    if not old_str:
        return None, "old_str must not be empty."
    count = content.count(old_str)
    if count == 0:
        return None, "No replacement was performed, old_str did not appear verbatim in the file."
    if count > 1:
        lines = [i + 1 for i, line in enumerate(content.splitlines()) if old_str.splitlines()[0] in line]
        return None, (
            f"No replacement was performed. old_str appears {count} times (lines {lines}); "
            "include more surrounding text so it matches exactly once, or rewrite the whole file "
            "(memory_write for a core file, write_topic_file for a topic file): repeated items are "
            "removed on every write."
        )
    return content.replace(old_str, new_str, 1), None


def keep_frontmatter(old: str, new: str) -> str:
    """``new``, with ``old``'s frontmatter put back if an edit dropped its description."""
    old_meta, _ = parse_frontmatter(old)
    new_meta, new_body = parse_frontmatter(new)
    if old_meta.get("description") and not new_meta.get("description"):
        return render_frontmatter(old_meta, new_body)
    return new
