import io
from pathlib import Path
from typing import Any, List, Optional, Union

from pydantic import model_validator

from gwenflow.logger import logger
from gwenflow.readers.base import Reader
from gwenflow.types import Document


class ExcelReader(Reader):

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Any) -> Any:
        missing = []
        for package in ("pandas", "openpyxl", "tabulate"):
            try:
                __import__(package)
            except ImportError:
                missing.append(package)
        if missing:
            raise ImportError(
                f"Missing required packages: {', '.join(missing)}. "
                f"Install with: `uv add {' '.join(missing)}`"
            )
        return values

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
                df = pd.read_excel(xls, sheet_name=sheet)
                truncated = max_rows is not None and len(df) > max_rows
                display_df = df.head(max_rows) if truncated else df
                documents.append(
                    Document(
                        id=self.key(f"{filename}_{sheet}"),
                        content=display_df.to_markdown(index=False),
                        metadata={"filename": filename, "sheet": sheet, "page": page_num, "rows": len(df), "columns": list(df.columns), "truncated": truncated},
                    )
                )
            return documents
        except Exception as e:
            logger.exception(f"Error reading file: {e}")
            return []
