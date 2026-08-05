import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

from gwenflow.logger import logger
from gwenflow.readers.base import Reader
from gwenflow.types import Document


@dataclass
class JSONReader(Reader):
    def get_page_content(self, page_data: Any) -> tuple[str, dict]:
        if not isinstance(page_data, dict):
            return str(page_data), {}
        page_data = dict(page_data)
        content = page_data.pop("content", None)
        if content is None:
            return json.dumps(page_data, ensure_ascii=False), page_data
        return str(content), page_data

    def read(self, file: Path) -> List[Document]:
        try:
            filename = self.get_file_name(file)
            json_data = json.loads(self.get_file_content(file, text_mode=True))
        except Exception as e:
            logger.error(f"Error reading file: {e}")
            return []

        if isinstance(json_data, dict):
            json_data = [json_data]

        documents = []
        for page_num, page_data in enumerate(json_data, start=1):
            try:
                content, extra_metadata = self.get_page_content(page_data)
            except Exception as e:
                logger.warning(f"Skipping entry {page_num} of {filename}: {e}")
                continue
            metadata = {"filename": filename, "page": page_num}
            metadata.update(extra_metadata)
            documents.append(
                Document(
                    id=self.key(f"{filename}_{page_num}"),
                    content=content,
                    metadata=metadata,
                )
            )
        return documents
