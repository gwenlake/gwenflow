import importlib.metadata

try:
    __version__ = importlib.metadata.version(__package__)
except importlib.metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""


from gwenflow.llms import ChatOpenAI, ChatAzureOpenAI, ChatGwenlake
from gwenflow.agents import Agent
from gwenflow.tasks import Task

__all__ = [
    "ChatOpenAI",
    "ChatAzureOpenAI",
    "ChatGwenlake",
    "Agent",
    "Task",
]