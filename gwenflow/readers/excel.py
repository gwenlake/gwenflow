import io
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

from gwenflow.logger import logger
from gwenflow.readers.base import Reader
from gwenflow.types import Document


@dataclass
class ExcelReader(Reader):
    def __post_init__(self) -> None:
        missing = [p for p in ("pandas", "openpyxl", "tabulate") if not __import__("importlib").util.find_spec(p)]
        if missing:
            raise ImportError(
                f"Missing required packages: {', '.join(missing)}. Install with: `uv add {' '.join(missing)}`"
            )

    def read(
        self,
        file: Union[Path, io.BytesIO],
        sheet_name: Optional[str] = None,
        max_rows: Optional[int] = None,
    ) -> List[Document]:
        import pandas as pd

        try:
            filename = self.get_file_name(file)
            content = self.get_file_content(file)
            xls = pd.ExcelFile(content)
            sheet_names = [sheet_name] if sheet_name is not None else xls.sheet_names
            documents = []
            for page_num, sheet in enumerate(sheet_names, start=1):
                dataframe = pd.read_excel(xls, sheet_name=sheet)
                truncated = max_rows is not None and len(dataframe) > max_rows
                display_df = dataframe.head(max_rows) if truncated else dataframe
                documents.append(
                    Document(
                        id=self.key(f"{filename}_{sheet}"),
                        content=display_df.to_markdown(index=False),
                        metadata={
                            "filename": filename,
                            "sheet": sheet,
                            "page": page_num,
                            "rows": len(dataframe),
                            "columns": list(dataframe.columns),
                            "truncated": truncated,
                        },
                    )
                )
            return documents
        except Exception as e:
            logger.exception(f"Error reading file: {e}")
            return []
