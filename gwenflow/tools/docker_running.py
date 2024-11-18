import traceback

from pydantic import BaseModel, Field
from typing import Optional, Dict
import re
from pathlib import Path
import uuid
import docker




class DockerRunner:
    def __init__(self, image: str = 'python:3.11-slim', work_dir: str = './coding'):
        self.client = docker.from_env()
        self.image = image
        self.id = None
        self.work_dir = Path(work_dir).resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.container = None

    def start(self, detach: bool = True):
        try:
            self.container = self.client.containers.run(
                image=self.image,
                command="/bin/sh",  # Keeps the container running
                name=str(uuid.uuid4()),  # Generate a unique name
                detach=detach,
                auto_remove=False,
                volumes={
                    str(self.work_dir): {"bind": "/work", "mode": "rw"}  # Properly formatted volumes
                },
            )
            self.id = self.container.id  # Save the container ID
            print(f"Container {self.id} started in detached mode.")
        except docker.errors.ContainerError as e:
            print(f"Error: {e}")
        except docker.errors.ImageNotFound:
            print("Error: Specified Docker image not found.")
        except docker.errors.APIError as e:
            print(f"API Error: {e}")
        except Exception as e:
            print(f"Unexpected Error: {e}\n{traceback.format_exc()}")

    def connect_to_running_container(self):
        """Attach to a running container if one exists."""
        try:
            containers = self.client.containers.list(filters={"status": "running"})
            for container in containers:
                if self.image in container.image.tags:
                    self.container = container
                    self.id = container.id
                    print(f"Connected to running container {self.id}.")
                    return True
            return False
        except Exception as e:
            print(f"Unexpected Error: {e}\n{traceback.format_exc()}")
            return False

    def is_container_running(self) -> bool:
        try:
            if self.container is None:
                return False
            self.container.reload()  # Refresh container status
            return self.container.status == 'running'
        except Exception:
            return False

    def run_file(self, file_path: str, language: str) -> str:
        try:
            if not self.is_container_running():
                return "Error: Container is not running."

            file_name = Path(file_path).name
            full_container_path = f"/work/{file_name}"
            self.container.put_archive(
                "/work",
                self._tar_file(file_path)
            )

            exec_command = self.container.exec_run(
                cmd=[language, full_container_path],
                stdout=True,
                stderr=True,
            )
            return exec_command.output.decode("utf-8")
        except Exception as e:
            return f"Unexpected Error: {e}\n{traceback.format_exc()}"

    def _tar_file(self, file_path: str):
        """Helper function to create a tar archive for the file."""
        import tarfile
        from io import BytesIO

        tar_stream = BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            tar.add(file_path, arcname=Path(file_path).name)
        tar_stream.seek(0)
        return tar_stream

if __name__ == "__main__":
    text = """
    ```bash
    pip install pandas yfinance matplotlib openpyxl
    ```
    
    ```python
    pip install pandas yfinance matplotlib openpyxl
    ```
ms
    """

    # code_blocks = extract_all_code_and_languages(text)
    # print(code_blocks)
