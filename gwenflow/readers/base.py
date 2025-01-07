from typing import Any, List, Union
from pydantic import BaseModel, Field, ConfigDict

import io
import requests
from typing import List
from pathlib import Path

from gwenflow.utils import logger
from gwenflow.types import Document
from gwenflow.utils.aws import aws_s3_read_file, aws_s3_read_text_file, aws_s3_uri_to_bucket_key


class Reader(BaseModel):

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def read(self, obj: Any) -> List[Document]:
        raise NotImplementedError

    def get_file_name(self, file: Union[Path, io.BytesIO]):
        if isinstance(file, io.BytesIO):
            return "noname"
        if not isinstance(file, Path):
            return str(Path(file))
        return str(file)

    def get_file_content(self, file: Union[Path, io.BytesIO], text_mode: bool=False):

        try:

            if isinstance(file, io.BytesIO):
                return file

            filename = str(file)

            if filename.startswith("s3://"):
                bucket, key = aws_s3_uri_to_bucket_key(file)
                if text_mode:
                    return aws_s3_read_text_file(bucket, key)
                return aws_s3_read_file(bucket, key)

            elif filename.startswith("http://") or filename.startswith("https://"):
                response = requests.get(str(file))
                if text_mode:
                    return response.text
                return io.BytesIO(response.content)
            
            else:
                if not isinstance(file, Path):
                    file = Path(file)
                if not file.exists():
                    raise FileNotFoundError(f"Could not find file: {file}")
                if text_mode:
                    return file.read_text("utf-8")
                return io.BytesIO(file.read_bytes())
        
        except Exception as e:
            logger.error(f"Error reading file: {e}")
        
        return None