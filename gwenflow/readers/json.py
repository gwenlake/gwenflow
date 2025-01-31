import json
from pathlib import Path
from typing import List

from gwenflow.readers.base import Reader
from gwenflow.types import Document
from gwenflow.utils import logger


class JSONReader(Reader):

    def read(self, file: Path) -> List[Document]:

        try:

            filename = self.get_file_name(file)
            content = self.get_file_content(file, text_mode=True)

            json_data = json.loads(content)

            if isinstance(json_data, dict):
                json_data = [json_data]

            documents = []
            for page_num, page_data in enumerate(json_data, start=1):
                content = page_data.pop("content")
                metadata = {"filename": filename, "page": page_num}
                metadata.update(page_data)
                documents.append(
                    Document(
                        id=self.key(f"{filename}_{page_num}"),
                        content=content,
                        metadata=metadata,
                    )
                )

        except Exception as e:
            logger.error(f"Error reading file: {e}")
            return []

        return documents
