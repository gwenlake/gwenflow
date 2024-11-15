import importlib.metadata

try:
    __version__ = importlib.metadata.version(__package__)
except importlib.metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""


from gwenflow.llms import ChatGwenlake, ChatOpenAI, ChatAzureOpenAI, ChatOllama
from gwenflow.agents import Agent
from gwenflow.tasks import Task
from gwenflow.tools import Tool
from gwenflow.flows import Flow, AutoFlow


__all__ = [
    "ChatGwenlake",
    "ChatOpenAI",
    "ChatAzureOpenAI",
    "ChatOllama",
    "Agent",
    "Task",
    "Tool",
    "Flow",
    "AutoFlow",
]