from setuptools import setup, find_packages

setup(
    name="gwenflow",
    description = "Gwenflow is a framework for your LLM applications.",
    version="0.2.1",
    url="https://github.com/gwenlake/gwenflow",
    author="The Gwenlake Team",
    author_email="info@gwenlake.com",
    install_requires=["httpx", "pydantic", "pyyaml", "numpy", "pandas", "tiktoken"],
    packages=find_packages(exclude=("tests")),
    python_requires=">=3.11",
)