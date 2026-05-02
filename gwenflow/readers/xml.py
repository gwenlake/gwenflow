import io
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

from gwenflow.logger import logger
from gwenflow.readers.base import Reader
from gwenflow.types import Document


@dataclass
class XmlReader(Reader):
    def __post_init__(self) -> None:
        try:
            __import__("lxml")
        except ImportError:
            raise ImportError("Missing required package: lxml. Install with: `uv add lxml`")

    def read(self, file: Union[Path, io.BytesIO]) -> List[Document]:
        from lxml import etree

        try:
            filename = self.get_file_name(file)
            content = self.get_file_content(file)
            tree = etree.parse(content)
            root = tree.getroot()
            xml_text = etree.tostring(root, method="text", encoding="unicode", with_tail=False)
            clean_text = "\n".join(line.strip() for line in xml_text.splitlines() if line.strip())
            return [
                Document(
                    id=self.key(f"{filename}_xml"),
                    content=clean_text,
                    metadata={"filename": filename, "root_tag": root.tag},
                )
            ]
        except Exception as e:
            logger.exception(f"Error reading XML file: {e}")
            return []
