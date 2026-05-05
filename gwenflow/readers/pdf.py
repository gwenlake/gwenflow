import io
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

from gwenflow.logger import logger
from gwenflow.readers.base import Reader
from gwenflow.types import Document


@dataclass
class PDFReader(Reader):
    def read(self, file: Union[Path, io.BytesIO]) -> List[Document]:
        try:
            import pdfplumber
        except ImportError as e:
            raise ImportError("pdfplumber is not installed. Please install it with `pip install pdfplumber`.") from e

        try:
            filename = self.get_file_name(file)
            content = self.get_file_content(file)
            pdf_file = content if isinstance(content, io.BytesIO) else io.BytesIO(content)
            documents = []
            with pdfplumber.open(pdf_file) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    safe_text = text.encode("utf-8", errors="ignore").decode("utf-8")
                    documents.append(
                        Document(
                            id=self.key(f"{filename}_{i + 1}"),
                            content=safe_text,
                            metadata={"filename": filename, "page": i + 1, "tables": page.extract_tables()},
                        )
                    )
        except Exception as e:
            logger.error(f"Error reading file: {e}")
            return []
        return documents
