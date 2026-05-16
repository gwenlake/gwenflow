from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
from uuid import uuid4

from pydantic import Field
from pydantic.fields import FieldInfo

from gwenflow.logger import logger
from gwenflow.tools.tool import BaseTool


def _unwrap(value, fallback):
    if isinstance(value, FieldInfo):
        return value.default if value.default is not None else fallback
    return value


@dataclass(kw_only=True)
class LocalFileSystemBase(BaseTool):
    target_directory: Optional[Union[Path, str]] = None
    default_extension: str = "txt"

    def __post_init__(self) -> None:
        self.target_directory = Path(self.target_directory).resolve() if self.target_directory else Path.cwd().resolve()
        self.target_directory.mkdir(parents=True, exist_ok=True)
        self.default_extension = self.default_extension.lstrip(".")
        super().__post_init__()


@dataclass(kw_only=True)
class LocalFileWriteTool(LocalFileSystemBase):
    name: str = "LocalFileWriteTool"
    description: str = (
        "Write content to a local file under the target directory. "
        "If filename is omitted, a UUID-based name is generated. "
        "If filename has no extension, default_extension is appended."
    )

    def _run(
        self,
        content: str = Field(description="Content to write to the file."),
        filename: str = Field(
            description="Name of the file. Empty = generate a UUID name.",
            default="",
        ),
        directory: str = Field(
            description="Directory relative to target_directory (or absolute). Empty = target_directory.",
            default="",
        ),
        extension: str = Field(
            description="File extension (without dot). Empty = default_extension or filename's suffix.",
            default="",
        ),
    ) -> str:
        filename = _unwrap(filename, "") or str(uuid4())
        directory = _unwrap(directory, "")
        extension = _unwrap(extension, "")

        if "." in filename:
            stem_path = Path(filename)
            filename = stem_path.stem
            extension = extension or stem_path.suffix.lstrip(".")
        extension = (extension or self.default_extension).lstrip(".")

        if directory:
            dir_candidate = Path(directory)
            dir_path = dir_candidate if dir_candidate.is_absolute() else self.target_directory / dir_candidate
        else:
            dir_path = self.target_directory
        dir_path = dir_path.resolve()

        try:
            dir_path.relative_to(self.target_directory)
        except ValueError:
            return f"Error: Directory '{directory}' is outside target_directory"

        dir_path.mkdir(parents=True, exist_ok=True)
        file_path = dir_path / f"{filename}.{extension}"

        try:
            file_path.write_text(content, encoding="utf-8")
        except OSError as e:
            logger.error(f"Failed to write file: {e}")
            return f"Error: failed to write file: {e}"

        logger.info(f"Wrote {file_path}")
        return f"Successfully wrote file to: {file_path}"


@dataclass(kw_only=True)
class LocalFileReadTool(LocalFileSystemBase):
    name: str = "LocalFileReadTool"
    description: str = "Read text content from a local file under the target directory."

    def _run(
        self,
        filename: str = Field(description="Filename (with extension) to read."),
        directory: str = Field(
            description="Directory relative to target_directory (or absolute). Empty = target_directory.",
            default="",
        ),
    ) -> str:
        directory = _unwrap(directory, "")

        if directory:
            dir_candidate = Path(directory)
            dir_path = dir_candidate if dir_candidate.is_absolute() else self.target_directory / dir_candidate
        else:
            dir_path = self.target_directory

        file_path = (dir_path / filename).resolve()
        try:
            file_path.relative_to(self.target_directory)
        except ValueError:
            return f"Error: File '{filename}' is outside target_directory"

        if not file_path.exists():
            return f"File not found: {file_path}"
        if not file_path.is_file():
            return f"Error: Not a file: {file_path}"

        try:
            return file_path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, PermissionError, OSError) as e:
            return f"Error reading file: {e}"
