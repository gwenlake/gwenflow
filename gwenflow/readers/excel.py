import io
from pathlib import Path
from typing import Any, List, Optional, Union

from pydantic import model_validator

from gwenflow.logger import logger
from gwenflow.readers.base import Reader
from gwenflow.types import Document


class ExcelReader(Reader):
    sheet_name: Union[str, None] = None
    sep: str = ","
    decimal: str = "."
    max_rows: Optional[int] = None

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

    def _build_document(self, df: Any, id: str, **metadata) -> Document:
        truncated = self.max_rows is not None and len(df) > self.max_rows
        display_df = df.head(self.max_rows) if truncated else df
        return Document(
            id=id,
            content=display_df.to_markdown(index=False),
            metadata={**metadata, "rows": len(df), "columns": list(df.columns), "truncated": truncated},
        )

    def _read_csv(self, file: Union[Path, io.BytesIO], filename: str) -> List[Document]:
        import pandas as pd
        text_content = self.get_file_content(file, text_mode=True)
        source = io.StringIO(text_content) if isinstance(text_content, str) else text_content
        df = pd.read_csv(source, sep=self.sep, decimal=self.decimal)
        return [self._build_document(df, id=self.key(filename), filename=filename)]

    def _read_excel(self, file: Union[Path, io.BytesIO], filename: str) -> List[Document]:
        import pandas as pd
        content = self.get_file_content(file)
        xls = pd.ExcelFile(content)
        sheet_names = [self.sheet_name] if self.sheet_name is not None else xls.sheet_names
        documents = []
        for page_num, sheet in enumerate(sheet_names, start=1):
            df = pd.read_excel(xls, sheet_name=sheet)
            documents.append(self._build_document(
                df,
                id=self.key(f"{filename}_{sheet}"),
                filename=filename,
                sheet=sheet,
                page=page_num,
            ))
        return documents

    def read(self, file: Union[Path, io.BytesIO]) -> List[Document]:
        try:
            filename = self.get_file_name(file)
            is_csv = isinstance(filename, str) and filename.lower().endswith(".csv")
            return self._read_csv(file, filename) if is_csv else self._read_excel(file, filename)
        except Exception as e:
            logger.exception(f"Error reading file: {e}")
            return []