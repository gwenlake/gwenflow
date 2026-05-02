import io
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

from gwenflow.logger import logger
from gwenflow.readers.base import Reader
from gwenflow.types import Document


@dataclass
class CSVReader(Reader):
    def __post_init__(self) -> None:
        missing = [p for p in ("pandas", "tabulate") if not __import__("importlib").util.find_spec(p)]
        if missing:
            raise ImportError(
                f"Missing required packages: {', '.join(missing)}. Install with: `uv add {' '.join(missing)}`"
            )

    def read(
        self,
        file: Union[Path, io.BytesIO],
        sep: str = ",",
        decimal: str = ".",
        max_rows: Optional[int] = None,
    ) -> List[Document]:
        import pandas as pd

        try:
            filename = self.get_file_name(file)
            text_content = self.get_file_content(file, text_mode=True)
            source = io.StringIO(text_content) if isinstance(text_content, str) else text_content
            dataframe = pd.read_csv(source, sep=sep, decimal=decimal)
            truncated = max_rows is not None and len(dataframe) > max_rows
            display_df = dataframe.head(max_rows) if truncated else dataframe
            return [
                Document(
                    id=self.key(filename),
                    content=display_df.to_markdown(index=False),
                    metadata={
                        "filename": filename,
                        "rows": len(dataframe),
                        "columns": list(dataframe.columns),
                        "truncated": truncated,
                    },
                )
            ]
        except Exception as e:
            logger.exception(f"Error reading file: {e}")
            return []
