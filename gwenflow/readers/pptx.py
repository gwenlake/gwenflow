import io
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

from gwenflow.logger import logger
from gwenflow.readers.base import Reader
from gwenflow.types import Document


@dataclass
class PptxReader(Reader):
    def __post_init__(self) -> None:
        try:
            __import__("pptx")
        except ImportError:
            raise ImportError("Missing required package: python-pptx. Install with: `uv add python-pptx`")

    def iter_shapes(self, shapes):
        from pptx.enum.shapes import MSO_SHAPE_TYPE

        for shape in shapes:
            if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
                yield from self.iter_shapes(shape.shapes)
            else:
                yield shape

    def read(self, file: Union[Path, io.BytesIO]) -> List[Document]:
        from pptx import Presentation

        try:
            filename = self.get_file_name(file)
            content = self.get_file_content(file)
            prs = Presentation(content)
            documents = []
            for slide_num, slide in enumerate(prs.slides, start=1):
                texts = []
                tables = []
                try:
                    for shape in self.iter_shapes(slide.shapes):
                        if shape.has_table:
                            tables.append([[cell.text.strip() for cell in row.cells] for row in shape.table.rows])
                            continue
                        if not shape.has_text_frame:
                            continue
                        for para in shape.text_frame.paragraphs:
                            line = "".join(run.text for run in para.runs).strip()
                            if line:
                                texts.append(line)
                except Exception as e:
                    logger.warning(f"Skipping slide {slide_num} of {filename}: {e}")
                    continue

                slide_text = "\n".join(texts)
                table_text = self.format_tables(tables)
                if table_text:
                    slide_text = f"{slide_text}\n\n{table_text}" if slide_text else table_text

                documents.append(
                    Document(
                        id=self.key(f"{filename}_slide{slide_num}"),
                        content=slide_text,
                        metadata={"filename": filename, "page": slide_num, "slide": slide_num, "tables": tables},
                    )
                )
            return documents
        except Exception as e:
            logger.exception(f"Error reading file: {e}")
            return []
