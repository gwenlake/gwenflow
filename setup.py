from setuptools import setup, find_packages

setup(
    name="gwenflow",
    description="A framework for orchestrating applications powered by autonomous AI agents and LLMs.",
    version="0.4.5",
    url="https://github.com/gwenlake/gwenflow",
    author="The Gwenlake Team",
    author_email="info@gwenlake.com",
    install_requires=[
        "httpx",
        "pydantic",
        "tqdm",
        "pyyaml",
        "beautifulsoup4",
        "fsspec",
        "rich",
        "tiktoken",
        "openai",
        "langchain",
        "pyarrow",
        "lancedb",
        "qdrant-client",
    ],
    packages=find_packages(exclude=("tests")),
    python_requires=">=3.11",
)
