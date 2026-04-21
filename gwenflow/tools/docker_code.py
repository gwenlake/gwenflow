import docker
from pydantic import Field

from gwenflow.tools import BaseTool
from gwenflow.logger import logger


DESCRIPTION = """\
This function to runs Python code in the current environment.
If successful, returns the value of `variable_to_return` if provided otherwise returns a success message.
If failed, returns an error message.
Returns the value of `variable_to_return` if successful, otherwise returns an error message.
"""


class DockerCodeTool(BaseTool):

    name: str = "DockerCodeTool"
    description: str = DESCRIPTION
    
    def _run(self,
        code: str = Field(description="The code to run."),
        language: str = Field(description="Language (sh, bash or python).", default=None),
    ):
        
        if language == "python":
            image = "python:3.12-alpine"
            command = ["python", "-c", code]

        elif language == "bash":
            image = "alpine:latest"
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
                tmpfs={'/tmp': 'size=50m'}
            )

            logger.debug(f"Command result: {result.decode('utf-8').strip()}")

            return result.decode('utf-8').strip()
            
        except docker.errors.ContainerError as e:
            return f"Docker error:\n{e.stderr.decode('utf-8')}"
        
        except Exception as e:
            return f"Error: {str(e)}"
