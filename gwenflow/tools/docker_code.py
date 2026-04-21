import docker
from pydantic import Field

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

    def _run(
        self,
        code: str = Field(description="The code to run."),
        language: str = Field(description="Language (sh or python).", default=None),
    ):
        if language == "python":
            image = "python:3.12-alpine"
            command = ["python", "-c", code]

        elif language == "sh":
            image = "curlimages/curl"
            command = ["sh", "-c", code]

        else:
            return "Unsupported language."

        try:
            logger.debug(f"Running {language} code:\n\n{code}\n\n")

            client = docker.from_env()
            result = client.containers.run(
                image=image,
                command=command,
                remove=True,
                network_disabled=False,
                mem_limit="128m",
                nano_cpus=500000000,
                read_only=False,
                user="nobody",
                security_opt=["no-new-privileges:true"],
                tmpfs={"/tmp": "size=50m"},
            )

            logger.debug(f"Command result: {result.decode('utf-8').strip()}")

            return result.decode("utf-8").strip()

        except docker.errors.ContainerError as e:
            return f"Docker error:\n{e.stderr.decode('utf-8')}"

        except Exception as e:
            return f"Error: {str(e)}"
