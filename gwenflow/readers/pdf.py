
import io
from typing import List, Union
from pathlib import Path

from gwenflow.logger import logger
from gwenflow.types import Document
from gwenflow.readers.base import Reader


class PDFReader(Reader):

    def read(self, file: Union[Path, io.BytesIO]) -> List[Document]:

        try:
            import pdfplumber
        except ImportError:
            raise ImportError("pdfplumber is not installed. Please install it with `pip install pdfplumber`.")

        try:
            filename = self.get_file_name(file)
            content = self.get_file_content(file)

            if isinstance(content, io.BytesIO):
                pdf_file = content
            else:
                pdf_file = io.BytesIO(content)

            documents = []
            with pdfplumber.open(pdf_file) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    safe_text = text.encode('utf-8', errors='ignore').decode('utf-8')

                    tables = page.extract_tables()

                    metadata = dict(
                        filename=filename,
                        page=i + 1,
                        tables=tables,
                    )

                    doc = Document(
                        id=self.key(f"{filename}_{i + 1}"),
                        content=safe_text,
                        metadata=metadata
                    )
                    documents.append(doc)

        except Exception as e:
            logger.error(f"Error reading file: {e}")
            return []

        return documents