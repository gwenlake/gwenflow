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
        try:
            import pandas # noqa: F401
        except ImportError:
            raise ImportError("`pandas` is not installed. Please install it with `uv add pandas`.")
        try:
            import openpyxl # noqa: F401
        except ImportError:
            raise ImportError("`openpyxl` is not installed. Please install it with `uv add openpyxl`.")
        try:
            import tabulate # noqa: F401
        except ImportError:
            raise ImportError("`tabulate` is not installed. Please install it with `uv add tabulate`.")
        return values

    def _df_to_text(self, df) -> str:
        if self.max_rows is not None and len(df) > self.max_rows:
            df = df.head(self.max_rows)
        return df.to_markdown(index=False)

    def read(self, file: Union[Path, io.BytesIO]) -> List[Document]:
        import pandas as pd
        try:
            filename = self.get_file_name(file)

            name = filename.lower() if isinstance(filename, str) else ""
            is_csv = name.endswith(".csv")

            documents = []

            if is_csv:
                text_content = self.get_file_content(file, text_mode=True)
                df = pd.read_csv(io.StringIO(text_content) if isinstance(text_content, str) else text_content, sep=self.sep, decimal=self.decimal)
                documents.append(
                    Document(
                        id=self.key(f"{filename}"),
                        content=self._df_to_text(df),
                        metadata={"filename": filename, "rows": len(df), "columns": list(df.columns)},
                    )
                )
            else:
                content = self.get_file_content(file)
                xls = pd.ExcelFile(content)
                sheet_names = [self.sheet_name] if self.sheet_name is not None else xls.sheet_names
                for page_num, sheet in enumerate(sheet_names, start=1):
                    df = pd.read_excel(xls, sheet_name=sheet)
                    documents.append(
                        Document(
                            id=self.key(f"{filename}_{sheet}"),
                            content=self._df_to_text(df),
                            metadata={
                                "filename": filename,
                                "sheet": sheet,
                                "page": page_num,
                                "rows": len(df),
                                "columns": list(df.columns),
                            },
                        )
                    )

        except Exception as e:
            logger.exception(f"Error reading file: {e}")
            return []

        return documents
