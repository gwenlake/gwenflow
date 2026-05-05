import asyncio
import inspect
import json
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel
from tenacity import RetryError, retry, stop_after_attempt, wait_exponential

from gwenflow.logger import logger


def _parse_docstring(func) -> tuple[str, dict[str, str]]:
    """Extract the summary and per-parameter descriptions from a docstring.

    Supports Google style (Args:) and reStructuredText style (:param name:).
    Returns (summary, {param_name: description}).
    """
    docstring = inspect.getdoc(func)
    if not docstring:
        return "", {}

    lines = docstring.splitlines()

    # --- RST style: :param name: description ---
    rst_params: dict[str, str] = {}
    for line in lines:
        m = re.match(r"\s*:param\s+(\w+)\s*:\s*(.*)", line)
        if m:
            rst_params[m.group(1)] = m.group(2).strip()

    # --- Summary: everything before the first section header or :param ---
    section_re = re.compile(r"^(\w[\w\s]*):\s*$")
    summary_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if section_re.match(stripped) or re.match(r":param\s", stripped):
            break
        summary_lines.append(stripped)
    summary = " ".join(line for line in summary_lines if line).strip()

    if rst_params:
        return summary, rst_params

    # --- Google style: Args: / Parameters: section ---
    args_headers = {"args", "arguments", "parameters"}
    other_headers = re.compile(r"^(\w[\w\s]*):\s*$")
    param_line = re.compile(r"^[ \t]{2,}(\w+)(?:\s*\([^)]*\))?\s*:\s*(.*)")

    google_params: dict[str, str] = {}
    in_args = False
    current_param: str | None = None
    current_desc: list[str] = []

    for line in lines:
        stripped = line.strip()
        section_m = other_headers.match(stripped)
        if section_m:
            if current_param:
                google_params[current_param] = " ".join(current_desc).strip()
                current_param, current_desc = None, []
            in_args = section_m.group(1).lower() in args_headers
            continue

        if not in_args:
            continue

        param_m = param_line.match(line)
        if param_m:
            if current_param:
                google_params[current_param] = " ".join(current_desc).strip()
            current_param = param_m.group(1)
            current_desc = [param_m.group(2).strip()] if param_m.group(2).strip() else []
        elif current_param and stripped:
            current_desc.append(stripped)

    if current_param:
        google_params[current_param] = " ".join(current_desc).strip()

    return summary, google_params


def function_to_json_schema(func, name: str = None, description: str = None) -> dict:
    """Convert a Python function into an OpenAI-compatible tool schema.

    Parameter descriptions are resolved in priority order:
      1. pydantic Field(description=...) as the parameter default
      2. Docstring Args / :param section
    The function description resolves as:
      1. Explicit ``description`` argument
      2. Docstring summary (first paragraph)
      3. Raw ``func.__doc__``
    """
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }

    try:
        signature = inspect.signature(func)
    except ValueError as e:
        raise ValueError(f"Failed to get signature for function {func.__name__}: {str(e)}") from e

    doc_summary, doc_params = _parse_docstring(func)

    parameters = {}
    for param in signature.parameters.values():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        param_type = type_map.get(param.annotation, "string")

        # Priority: Field(description=...) > docstring > nothing
        if hasattr(param.default, "description"):
            param_desc = param.default.description
        else:
            param_desc = doc_params.get(param.name, "")

        entry: dict = {"type": param_type}
        if param_desc:
            entry["description"] = param_desc
        parameters[param.name] = entry

    required = [
        param.name
        for param in signature.parameters.values()
        if param.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        and (param.default is inspect.Parameter.empty or hasattr(param.default, "description"))
    ]

    return {
        "type": "function",
        "function": {
            "name": name or func.__name__,
            "description": description or doc_summary or func.__doc__ or "",
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required,
            },
        },
    }


@dataclass(kw_only=True)
class BaseTool(ABC):
    name: str = field(default=None)
    description: str = field(default=None)
    parameters: dict[str, Any] = field(default=None)
    function_schema: dict[str, Any] = field(default=None)
    max_results: int = 50
    max_retries: int | None = 3

    def __post_init__(self):
        self.function_schema = function_to_json_schema(self._run)
        if self.name is None:
            self.name = self.function_schema["function"]["name"]
        if self.description is None:
            self.description = self.function_schema["function"]["description"]
        if self.parameters is None:
            self.parameters = self.function_schema["function"]["parameters"]

    def _cast_response_to_str(self, response) -> str:
        if not response:
            return None
        if isinstance(response, str):
            return response
        elif isinstance(response, BaseModel):
            return response.model_dump_json(exclude_none=True)
        try:
            return json.dumps(response, ensure_ascii=False)
        except Exception as e:
            logger.error(e)
            return str(response)

    @abstractmethod
    def _run(self, **kwargs: Any) -> Any:
        """Actual implementation of the tool."""

    def run(self, **kwargs: Any) -> str:
        _retry = retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            reraise=True,
        )
        try:
            response = _retry(self._run)(**kwargs)
        except RetryError as e:
            logger.error(f"Tool {self.name} failed after {self.max_retries} retries: {e}")
            raise
        return self._cast_response_to_str(response)

    async def arun(self, **kwargs: Any) -> str:
        _retry = retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            reraise=True,
        )
        try:
            response = await asyncio.to_thread(_retry(self._run), **kwargs)
        except RetryError as e:
            logger.error(f"Tool {self.name} failed after {self.max_retries} retries: {e}")
            raise
        return self._cast_response_to_str(response)


@dataclass(kw_only=True, init=False)
class Tool(BaseTool):
    function: Callable[..., Any] = field(default=None)

    def __init__(
        self,
        function: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        parameters: dict[str, Any] | None = None,
        max_results: int = 50,
        max_retries: int | None = 3,
    ):
        self.function = function
        self.name = name
        self.description = description
        self.parameters = parameters
        self.function_schema = None
        self.max_results = max_results
        self.max_retries = max_retries
        self.__post_init__()

    def __post_init__(self):
        if self.function is not None:
            self.function_schema = function_to_json_schema(self.function)
            if self.name is None:
                self.name = self.function_schema["function"]["name"]
            if self.description is None:
                self.description = self.function_schema["function"]["description"]
            if self.parameters is None:
                self.parameters = self.function_schema["function"]["parameters"]
        else:
            super().__post_init__()

    def _run(self, **kwargs: Any) -> Any:
        return self.function(**kwargs)
