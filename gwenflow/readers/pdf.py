
import io
import requests
from typing import List
from pathlib import Path

from gwenflow.types import Document
from gwenflow.readers.base import Reader
from gwenflow.utils import logger


class PDFReader(Reader):

    def read(self, file: Path) -> List[Document]:

        try:
            import pymupdf
        except ImportError:
            raise ImportError("PyMuPDF is not installed. Please install it with `pip install PyMuPDF`.")

        try:
            filename = self.get_file_name(file)
            content  = self.get_file_content(file)

            documents = []
            for page in pymupdf.open(stream=content, filetype="pdf"):
                text = page.get_text()
                safe_text = text.encode('utf-8', errors='ignore').decode('utf-8')
                tables = []
                for table in page.find_tables():
                    tables.append(table.extract())
                filename = str(file)
                metadata = dict(filename=filename, page=page.number+1, tables=tables, images=[])
                doc = Document(id=f"{filename}_{page.number+1}", content=safe_text, metadata=metadata)
                documents.append(doc)

        except Exception as e:
            logger.error(f"Error reading file: {e}")
            return []

        return documents
