import json
import shlex
import subprocess
from pathlib import Path
from typing import Optional, Union

from pydantic import Field
from pydantic.fields import FieldInfo

from gwenflow.logger import logger
from gwenflow.tools import BaseTool


class ShellTool(BaseTool):
    name: str = "ShellTool"
    description: str = "Runs a shell command and returns the output or error."
    base_dir: Optional[Union[Path, str]] = None

    def _run(
        self,
        cmd: str = Field(description="The command to run."),
        tail: int = Field(description="The number of lines to return from the output.", default=100),
    ):
        if isinstance(tail, FieldInfo):
            tail = tail.default if tail.default is not None else 10

        try:
            logger.info(f"Running shell command: {cmd}")
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                cwd=str(self.base_dir) if self.base_dir else None,
            )

            if result.returncode != 0:
                return f"Error: {result.stderr}"

            output_lines = result.stdout.splitlines()
            tail_output = "\n".join(output_lines[-tail:]) if tail > 0 else result.stdout

            response_data = {
                "status": "success" if result.returncode == 0 else "error",
                "return_code": result.returncode,
                "stdout": tail_output,
                "stderr": result.stderr.strip(),
            }

            logger.debug(f"Command result: {response_data}")

            return json.dumps(response_data, ensure_ascii=False)

        except Exception as e:
            logger.warning(f"Failed to run shell command: {str(e)}")
            return f"Error: {e}"
