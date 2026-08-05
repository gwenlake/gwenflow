import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Union

import requests

from gwenflow.types import Document
from gwenflow.utils.aws import aws_s3_read_file, aws_s3_read_text_file, aws_s3_uri_to_bucket_key


@dataclass
class Reader:
    def key(self, text: str) -> str:
        return hashlib.md5(text.encode(), usedforsecurity=False).hexdigest()

    def read(self, obj: Any) -> List[Document]:
        raise NotImplementedError

    def get_file_name(self, file: Union[Path, io.BytesIO]) -> str:
        if isinstance(file, io.BytesIO):
            return "noname"
        if not isinstance(file, Path):
            return str(Path(file))
        return str(file)

    def get_file_content(self, file: Union[Path, io.BytesIO], text_mode: bool = False):
        if isinstance(file, io.BytesIO):
            return file.getvalue().decode("utf-8", errors="replace") if text_mode else file

        filename = str(file)

        if filename.startswith("s3://"):
            bucket, key = aws_s3_uri_to_bucket_key(file)
            if text_mode:
                return aws_s3_read_text_file(bucket, key)
            return aws_s3_read_file(bucket, key)

        elif filename.startswith("http://") or filename.startswith("https://"):
            response = requests.get(str(file))
            response.raise_for_status()
            if text_mode:
                return response.text
            return io.BytesIO(response.content)

        else:
            if not isinstance(file, Path):
                file = Path(file)
            if not file.exists():
                raise FileNotFoundError(f"Could not find file: {file}")
            if text_mode:
                return file.read_text("utf-8")
            return io.BytesIO(file.read_bytes())

        return None

    def format_tables(self, tables: List[List[List[Any]]]) -> str:
        blocks = []
        for table in tables:
            rows = [row for row in table if row]
            if not rows:
                continue
            lines = [
                "| " + " | ".join(str(cell).replace("\n", " ").strip() if cell else "" for cell in row) + " |"
                for row in rows
            ]
            lines.insert(1, "| " + " | ".join(["---"] * max(len(row) for row in rows)) + " |")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)
