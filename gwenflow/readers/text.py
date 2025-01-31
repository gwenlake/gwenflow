from pathlib import Path
from typing import List

from gwenflow.readers.base import Reader
from gwenflow.types import Document
from gwenflow.utils import logger


class TextReader(Reader):

    def read(self, file: Path) -> List[Document]:

        try:

            filename = self.get_file_name(file)
            content = self.get_file_content(file, text_mode=True)

            documents = [
                Document(
                    id=self.key(filename),
                    content=content,
                    metadata={"filename": filename},
                )
            ]

        except Exception as e:
            logger.error(f"Error reading file: {e}")
            return []

        return documents
