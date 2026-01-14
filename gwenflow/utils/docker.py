import shutil
import subprocess


def validate_docker_installation(agent_role: str) -> None:
    """Check if Docker is installed and running.

    Args:
        agent_role (str): The role of the agent requesting validation.

    Raises:
        RuntimeError: If Docker is missing or not running.
    """
    if not shutil.which("docker"):
        raise RuntimeError(
            f"Docker is not installed. Please install Docker to use code "
            f"execution with agent: {agent_role}"
        )

    try:
        subprocess.run(
            ["docker", "info"],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Docker is not running. Please start Docker to use code "
            f"execution with agent: {agent_role}"
        ) from e
