
import shutil
import subprocess

def validate_docker_installation() -> None:
    """Check if Docker is installed and running."""
    if not shutil.which("docker"):
        raise RuntimeError(
            f"Docker is not installed. Please install Docker to use code execution with agent: {self.role}"
        )

    try:
        subprocess.run(
            ["docker", "info"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError:
        raise RuntimeError(
            f"Docker is not running. Please start Docker to use code execution with agent: {self.role}"
        )