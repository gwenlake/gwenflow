import json
import inspect
import re

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from pydantic import BaseModel


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
    SECTION_RE = re.compile(r"^(\w[\w\s]*):\s*$")
    summary_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if SECTION_RE.match(stripped) or re.match(r":param\s", stripped):
            break
        summary_lines.append(stripped)
    summary = " ".join(l for l in summary_lines if l).strip()

    if rst_params:
        return summary, rst_params

    # --- Google style: Args: / Parameters: section ---
    ARGS_HEADERS = {"args", "arguments", "parameters"}
    OTHER_HEADERS = re.compile(r"^(\w[\w\s]*):\s*$")
    PARAM_LINE = re.compile(r"^[ \t]{2,}(\w+)(?:\s*\([^)]*\))?\s*:\s*(.*)")

    google_params: dict[str, str] = {}
    in_args = False
    current_param: str | None = None
    current_desc: list[str] = []

    for line in lines:
        stripped = line.strip()
        section_m = OTHER_HEADERS.match(stripped)
        if section_m:
            if current_param:
                google_params[current_param] = " ".join(current_desc).strip()
                current_param, current_desc = None, []
            in_args = section_m.group(1).lower() in ARGS_HEADERS
            continue

        if not in_args:
            continue

        param_m = PARAM_LINE.match(line)
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
class Tool(ABC):
    name: str | None = None
    description: str | None = None
    parameters: Optional[dict[str, Any]] = None
    tool_type: str = "function"
    max_results: int = 50

    def __post_init__(self) -> None:
        _schema = function_to_json_schema(self._run, name=self.name, description=self.description)
        if not self.name:
            self.name = _schema["function"]["name"]
        if not self.description:
            self.description = _schema["function"]["description"]
        if self.parameters is None:
            self.parameters = _schema["function"]["parameters"]

    def to_openai(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description or "",
                "parameters": self.parameters,
            },
        }

    def to_openai_new(self) -> dict:
        return {
            "type": "function",
            "name": self.name,
            "description": self.description or "",
            "parameters": self.parameters,
        }

    def _cast_response_to_str(self, response) -> str:
        if not response:
            return None
        if isinstance(response, str):
            return response
        elif isinstance(response, BaseModel):
            return response.model_dump_json(exclude_none=True)
        try:
            return json.dumps(response, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(response)

    @abstractmethod
    def _run(self, **kwargs: Any) -> Any:
        """Actual implementation of the tool."""

    def run(self, **kwargs: Any) -> str:
        response = self._run(**kwargs)
        return self._cast_response_to_str(response)

    async def arun(self, **kwargs: Any) -> str:
        import asyncio
        response = await asyncio.to_thread(self._run, **kwargs)
        return self._cast_response_to_str(response)
