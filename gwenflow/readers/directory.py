
from abc import ABC
from typing import cast
from pathlib import Path, PurePosixPath

import fsspec
from fsspec.implementations.local import LocalFileSystem
from tqdm import tqdm

from gwenflow.types import Document
from gwenflow.readers.json import JSONReader
from gwenflow.readers.pdf import PDFReader

default_file_reader_cls = {
    ".json": JSONReader,
    ".pdf": PDFReader,
}


def get_default_fs() -> fsspec.AbstractFileSystem:
    return LocalFileSystem()


class SimpleDirectoryReader(ABC):
    """SimpleDirectoryReader"""

    def __init__(
        self,
        input_dir: str | None = None,
        input_files: list | None = None,
        exclude: list | None = None,
        exclude_hidden: bool = True,
        errors: str = "ignore",
        recursive: bool = True,
        encoding: str = "utf-8",
        filename_as_id: bool = False,
        required_exts: list[str] | None = None,
        raise_on_error: bool = False,
    ) -> None:

        if not input_dir and not input_files:
            raise ValueError("Must provide either `input_dir` or `input_files`.")

        self.fs = get_default_fs()

        self.exclude = exclude
        self.exclude_hidden = exclude_hidden
        self.errors = errors
        self.recursive = recursive
        self.encoding = encoding
        self.filename_as_id = filename_as_id
        self.required_exts = required_exts
        self.raise_on_error = raise_on_error

        if input_files:
            self.input_files = []
            for path in input_files:
                if not self.fs.isfile(path):
                    raise ValueError(f"File {path} does not exist.")
                input_file = Path(path)
                self.input_files.append(input_file)

        elif input_dir:
            if not self.fs.isdir(input_dir):
                raise ValueError(f"Directory {input_dir} does not exist.")
            self.input_dir = Path(input_dir)
            self.exclude = exclude
            self.input_files = self._add_files(self.input_dir)

    @staticmethod
    def is_hidden(path: Path | PurePosixPath) -> bool:
        return any(
            part.startswith(".") and part not in [".", ".."] for part in path.parts
        )

    def _add_files(self, input_dir: Path | PurePosixPath) -> list[Path | PurePosixPath]:

        all_files = []

        if self.recursive:
            list_files = input_dir.rglob("*")
        else:
            list_files = input_dir.glob("*")
        
        # only keep required_exts
        for file in list_files:
            if self.required_exts:
                if file.suffix.lower() in self.required_exts:
                    all_files.append(file)
            else:
                all_files.append(file)

        return list(set(all_files)) # remove duplicates

    
    def read(self, show_progress: bool = False) -> list[Document]:

        documents = []

        files_to_process = self.input_files

        if show_progress:
            files_to_process = tqdm(self.input_files, desc="Reading files", unit="file")

        for input_file in files_to_process:
            if SimpleDirectoryReader.is_hidden(input_file):
                continue
            documents.extend(
                SimpleDirectoryReader.read_file(
                    input_file=input_file,
                    filename_as_id=self.filename_as_id,
                    encoding=self.encoding,
                    errors=self.errors,
                    raise_on_error=self.raise_on_error,
                )
            )
        return documents
    
    @staticmethod
    def read_file(
        input_file: Path | PurePosixPath,
        filename_as_id: bool = False,
        encoding: str = "utf-8",
        errors: str = "ignore",
        raise_on_error: bool = False,
    ) -> list[Document]:
        
        documents = []

        file_suffix = input_file.suffix.lower()
        if not file_suffix:
            return []
        
        # supported formats
        supported_suffix = list(default_file_reader_cls.keys())
        if file_suffix in supported_suffix:
            try:
                reader = default_file_reader_cls[file_suffix]()
                documents = reader.read(file=input_file)
            except Exception as e:
                if raise_on_error:
                    raise Exception("Error loading file") from e
                print(f"Failed to load file {input_file} with error: {e}. Skipping...", flush=True)
                return []
        
        # other formats
        else:
            fs = get_default_fs()
            with fs.open(input_file, errors=errors, encoding=encoding) as f:
                content = cast(bytes, f.read()).decode(encoding, errors=errors)

            doc = Document(content=content, name=str(input_file))  # type: ignore

            if filename_as_id:
                doc.id = str(input_file)

            documents.append(doc)

        return documents