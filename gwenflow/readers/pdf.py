
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

            documents = []
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    safe_text = text.encode('utf-8', errors='ignore').decode('utf-8')

                    tables = page.extract_tables()

                    images = []
                    for img in page.images:
                        images.append({
                            'x0': img['x0'],
                            'y0': img['y0'],
                            'x1': img['x1'],
                            'y1': img['y1'],
                            'name': img.get('name'),
                            'width': img.get('width'),
                            'height': img.get('height'),
                        })

                    metadata = dict(
                        filename=filename,
                        page=i + 1,
                        tables=tables,
                        images=images
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