import os
from pathlib import Path
from typing import Any, Optional, Union
from pydantic import Field, model_validator

from gwenflow.logger import logger
from gwenflow.tools import BaseTool

DESCRIPTION = """\
Executes Python or Shell (sh) code in an isolated Docker container.
Use this tool to perform computations, run scripts, or manipulate data.
You MUST use print() in Python or echo in Shell to output the results you want to see.
Returns the standard output (stdout) if successful, or the error message (stderr) if it fails.
"""


class DockerCodeTool(BaseTool):
    name: str = "DockerCodeTool"
    description: str = DESCRIPTION

    base_dir: Optional[Union[Path, str]] = None

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Any) -> Any:
        try:
            import docker  # noqa: F401
        except ImportError as e:
            raise ImportError("`docker` is not installed. Please install it with `uv add docker`.") from e
        return values

    def _run(
        self,
        code: str = Field(description="The code to run."),
        language: str = Field(description="Language (sh or python).", default=None),
    ):
        import docker

        try:
            logger.debug(f"Running {language} code:\n\n{code}\n\n")

            run_kwargs = {
                "image": None,
                "command": None,
                "remove": True,
                "network_disabled": False,
                "mem_limit": "128m",
                "nano_cpus": 500000000,
                "security_opt": ["no-new-privileges:true"],
                "tmpfs": {"/tmp": "size=50m"},
                "user": "nobody"
            }

            if language == "python":
                run_kwargs["image"] = "python:3.12-alpine"
                run_kwargs["command"] = ["python", "-c", code]
            elif language == "sh":
                run_kwargs["image"] = "alpine:latest"
                run_kwargs["command"] = ["sh", "-c", code]
            else:
                return "Unsupported language."

            if self.base_dir:
                abs_path = os.path.abspath(self.base_dir)
                os.makedirs(abs_path, exist_ok=True)                
                run_kwargs["volumes"] = {abs_path: {"bind": "/mnt/workspace", "mode": "rw"}}
                run_kwargs["working_dir"] = "/mnt/workspace"
                if os.name != 'nt':
                    run_kwargs["user"] = os.getuid()

            client = docker.from_env()
            result = client.containers.run(**run_kwargs)
            output = result.decode("utf-8").strip()

            logger.debug(f"Command result: {output}")

            return output

        except docker.errors.ContainerError as e:
            return f"Docker error:\n{e.stderr.decode('utf-8')}"

        except Exception as e:
            return f"Error: {str(e)}"
