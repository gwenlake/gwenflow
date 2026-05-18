import difflib
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

from pydantic import Field
from pydantic.fields import FieldInfo

from gwenflow.logger import logger
from gwenflow.tools.tool import BaseTool


def _default(value: Any, fallback: Any) -> Any:
    if isinstance(value, FieldInfo):
        return value.default if value.default is not None else fallback
    return value


@dataclass(kw_only=True)
class CodingBase(BaseTool):
    base_dir: Optional[Union[Path, str]] = None
    restrict_to_base_dir: bool = True
    max_lines: int = 2000
    max_bytes: int = 50_000

    def __post_init__(self) -> None:
        self.base_dir = Path(self.base_dir).resolve() if self.base_dir else Path.cwd().resolve()
        super().__post_init__()

    def _resolve_path(self, file_path: str) -> tuple[bool, Path]:
        candidate = Path(file_path)
        resolved = candidate.resolve() if candidate.is_absolute() else (self.base_dir / candidate).resolve()
        if self.restrict_to_base_dir:
            try:
                resolved.relative_to(self.base_dir)
            except ValueError:
                return False, resolved
        return True, resolved

    def _truncate(self, text: str) -> tuple[str, bool, int]:
        lines = text.split("\n")
        total = len(lines)
        truncated = False

        if total > self.max_lines:
            lines = lines[: self.max_lines]
            truncated = True

        result = "\n".join(lines)
        if len(result.encode("utf-8", errors="replace")) > self.max_bytes:
            kept: list[str] = []
            current = 0
            for line in lines:
                size = len((line + "\n").encode("utf-8", errors="replace"))
                if current + size > self.max_bytes:
                    break
                kept.append(line)
                current += size
            result = "\n".join(kept)
            truncated = True

        return result, truncated, total


@dataclass(kw_only=True)
class ReadFileTool(CodingBase):
    name: str = "ReadFileTool"
    description: str = (
        "Read the contents of a file with line numbers. Supports pagination via "
        "offset and limit for large files. Always read a file before editing it."
    )

    def _run(
        self,
        file_path: str = Field(description="Path to the file to read (relative to base_dir or absolute)."),
        offset: int = Field(description="Line number to start reading from (0-indexed).", default=0),
        limit: int = Field(description="Maximum number of lines to read. 0 uses max_lines.", default=0),
    ):
        offset = _default(offset, 0)
        limit = _default(limit, 0)
        safe, resolved = self._resolve_path(file_path)
        if not safe:
            return f"Error: Path '{file_path}' is outside the allowed base directory"
        if not resolved.exists():
            return f"Error: File not found: {file_path}"
        if not resolved.is_file():
            return f"Error: Not a file: {file_path}"

        try:
            with open(resolved, "rb") as f:
                if b"\x00" in f.read(8192):
                    return f"Error: Binary file detected: {file_path}"
        except OSError as e:
            return f"Error reading file: {e}"

        try:
            contents = resolved.read_text(encoding="utf-8", errors="replace")
        except (UnicodeDecodeError, PermissionError) as e:
            return f"Error reading file: {e}"

        if not contents:
            return f"File is empty: {file_path}"

        lines = contents.split("\n")
        total = len(lines)
        effective_limit = limit if limit and limit > 0 else self.max_lines
        selected = lines[offset : offset + effective_limit]

        width = max(len(str(offset + len(selected))), 4)
        numbered = "\n".join(f"{offset + i + 1:>{width}} | {line}" for i, line in enumerate(selected))
        output, truncated, _ = self._truncate(numbered)

        shown_end = offset + len(selected)
        if truncated or shown_end < total or offset > 0:
            output += f"\n[Showing lines {offset + 1}-{shown_end} of {total} total]"
        return output


@dataclass(kw_only=True)
class EditFileTool(CodingBase):
    name: str = "EditFileTool"
    description: str = (
        "Edit a file by replacing an exact text match with new text. The old_text "
        "must match exactly one location in the file (including whitespace and "
        "indentation). Returns a unified diff of the change."
    )

    def _run(
        self,
        file_path: str = Field(description="Path to the file to edit."),
        old_text: str = Field(description="Exact text to find and replace. Must match a unique location."),
        new_text: str = Field(description="Text to replace old_text with."),
    ):
        safe, resolved = self._resolve_path(file_path)
        if not safe:
            return f"Error: Path '{file_path}' is outside the allowed base directory"
        if not resolved.exists():
            return f"Error: File not found: {file_path}"
        if not resolved.is_file():
            return f"Error: Not a file: {file_path}"
        if not old_text:
            return "Error: old_text cannot be empty"
        if old_text == new_text:
            return "No changes needed: old_text and new_text are identical"

        try:
            contents = resolved.read_text(encoding="utf-8")
        except (UnicodeDecodeError, PermissionError) as e:
            return f"Error reading file: {e}"

        count = contents.count(old_text)
        if count == 0:
            return (
                f"Error: old_text not found in {file_path}. "
                "Make sure the text matches exactly (including whitespace and indentation)."
            )
        if count > 1:
            return (
                f"Error: old_text matches {count} locations in {file_path}. "
                "Provide more surrounding context to make the match unique."
            )

        new_contents = contents.replace(old_text, new_text, 1)
        try:
            resolved.write_text(new_contents, encoding="utf-8")
        except PermissionError as e:
            return f"Error writing file: {e}"

        diff_text = "".join(
            difflib.unified_diff(
                contents.splitlines(keepends=True),
                new_contents.splitlines(keepends=True),
                fromfile=f"a/{file_path}",
                tofile=f"b/{file_path}",
                n=3,
            )
        )
        if not diff_text:
            return "Edit applied but no visible diff generated"

        output, truncated, total = self._truncate(diff_text)
        if truncated:
            output += f"\n[Diff truncated: {total} lines total]"
        logger.info(f"Edited {file_path}")
        return output


@dataclass(kw_only=True)
class WriteFileTool(CodingBase):
    name: str = "WriteFileTool"
    description: str = (
        "Create a new file or overwrite an existing one entirely. "
        "Parent directories are created automatically. Prefer EditFileTool for "
        "modifying existing files."
    )

    def _run(
        self,
        file_path: str = Field(description="Path to the file to write."),
        contents: str = Field(description="Full contents to write to the file."),
    ):
        safe, resolved = self._resolve_path(file_path)
        if not safe:
            return f"Error: Path '{file_path}' is outside the allowed base directory"

        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(contents, encoding="utf-8")
        except PermissionError as e:
            return f"Error writing file: {e}"

        line_count = len(contents.split("\n"))
        logger.info(f"Wrote {file_path}")
        return f"Wrote {line_count} lines to {file_path}"


@dataclass(kw_only=True)
class GrepTool(CodingBase):
    name: str = "GrepTool"
    description: str = (
        "Search file contents for a pattern (regex). Returns matching lines with "
        "file paths and line numbers. Filter by file glob with `include` (e.g. '*.py')."
    )

    def _run(
        self,
        pattern: str = Field(description="Search pattern (regex by default)."),
        path: str = Field(description="Directory or file to search in. Empty = base_dir.", default=""),
        ignore_case: bool = Field(description="Case-insensitive search.", default=False),
        include: str = Field(description="Filter files by glob, e.g. '*.py'.", default=""),
        context: int = Field(description="Lines of context before/after each match.", default=0),
        limit: int = Field(description="Maximum number of matches.", default=100),
    ):
        path = _default(path, "")
        ignore_case = _default(ignore_case, False)
        include = _default(include, "")
        context = _default(context, 0)
        limit = _default(limit, 100)
        if not pattern:
            return "Error: Pattern cannot be empty"

        if path:
            safe, resolved = self._resolve_path(path)
            if not safe:
                return f"Error: Path '{path}' is outside the allowed base directory"
        else:
            resolved = self.base_dir
        if not resolved.exists():
            return f"Error: Path not found: {path or '.'}"

        cmd = ["grep", "-rn"]
        if ignore_case:
            cmd.append("-i")
        if context and context > 0:
            cmd.extend(["-C", str(context)])
        if include:
            cmd.extend(["--include", include])
        cmd.extend([pattern, str(resolved)])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(self.base_dir),
            )
        except subprocess.TimeoutExpired:
            return "Error: grep timed out after 30 seconds"
        except FileNotFoundError:
            return "Error: grep command not found"

        output = result.stdout
        if not output:
            if result.stderr:
                return f"Error: {result.stderr.strip()}"
            return f"No matches found for pattern: {pattern}"

        base_prefix = str(self.base_dir) + "/"
        output = output.replace(base_prefix, "")

        lines = output.split("\n")
        if limit and len(lines) > limit:
            output = "\n".join(lines[:limit]) + f"\n[Results limited to {limit} matches]"

        output, truncated, total = self._truncate(output)
        if truncated:
            output += f"\n[Output truncated: {total} lines total]"
        return output


@dataclass(kw_only=True)
class FindTool(CodingBase):
    name: str = "FindTool"
    description: str = (
        "Find files by glob pattern (e.g. '*.py', '**/*.json'). Returns matching "
        "file paths relative to the search directory."
    )

    def _run(
        self,
        pattern: str = Field(description="Glob pattern, e.g. '*.py' or '**/*.json'."),
        path: str = Field(description="Directory to search in. Empty = base_dir.", default=""),
        limit: int = Field(description="Maximum number of results.", default=500),
    ):
        path = _default(path, "")
        limit = _default(limit, 500)
        if not pattern:
            return "Error: Pattern cannot be empty"

        if path:
            safe, resolved = self._resolve_path(path)
            if not safe:
                return f"Error: Path '{path}' is outside the allowed base directory"
        else:
            resolved = self.base_dir
        if not resolved.exists():
            return f"Error: Path not found: {path or '.'}"
        if not resolved.is_dir():
            return f"Error: Not a directory: {path}"

        matches: list[str] = []
        for match in resolved.glob(pattern):
            try:
                rel = match.relative_to(self.base_dir)
            except ValueError:
                continue
            suffix = "/" if match.is_dir() else ""
            matches.append(str(rel) + suffix)
            if limit and len(matches) >= limit:
                break

        if not matches:
            return f"No files found matching pattern: {pattern}"

        result = "\n".join(sorted(matches))
        if limit and len(matches) >= limit:
            result += f"\n[Results limited to {limit} entries]"
        return result


@dataclass(kw_only=True)
class LsTool(CodingBase):
    name: str = "LsTool"
    description: str = (
        "List directory contents sorted alphabetically. Directories have a trailing '/'. " "Includes dotfiles."
    )

    def _run(
        self,
        path: str = Field(description="Directory to list. Empty = base_dir.", default=""),
        limit: int = Field(description="Maximum entries to return.", default=500),
    ):
        path = _default(path, "")
        limit = _default(limit, 500)
        if path:
            safe, resolved = self._resolve_path(path)
            if not safe:
                return f"Error: Path '{path}' is outside the allowed base directory"
        else:
            resolved = self.base_dir
        if not resolved.exists():
            return f"Error: Path not found: {path or '.'}"
        if not resolved.is_dir():
            return f"Error: Not a directory: {path}"

        try:
            iter_entries = sorted(resolved.iterdir(), key=lambda p: p.name.lower())
        except PermissionError:
            return f"Error: Permission denied: {path or '.'}"

        entries: list[str] = []
        for entry in iter_entries:
            suffix = "/" if entry.is_dir() else ""
            entries.append(entry.name + suffix)
            if limit and len(entries) >= limit:
                break

        if not entries:
            return f"Directory is empty: {path or '.'}"
        result = "\n".join(entries)
        if limit and len(entries) >= limit:
            result += f"\n[Listing limited to {limit} entries]"
        return result
