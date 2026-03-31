import inspect
import json
from contextlib import contextmanager
from typing import Any, Callable

from opentelemetry import baggage
from opentelemetry.context import attach, detach


def safe_serialize(obj: Any) -> str:
    try:
        if hasattr(obj, "model_dump_json") and callable(obj.model_dump_json):
            return obj.model_dump_json()
        if hasattr(obj, "dict") and callable(obj.dict):
            return json.dumps(obj.dict(), default=str)
        return str(obj)
    except Exception:
        return str(obj)


def extract_user_inputs(func: Callable, args: tuple, kwargs: dict) -> str:
    try:
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        arg_dict = dict(bound_args.arguments)

        arg_dict.pop("self", None)
        arg_dict.pop("cls", None)

        if "input" in arg_dict:
            return safe_serialize(arg_dict["input"])
        if "query" in arg_dict:
            return safe_serialize(arg_dict["query"])
        if "content" in arg_dict:
            return safe_serialize(arg_dict["content"])
        if "tool_call" in arg_dict:
            return safe_serialize(arg_dict["tool_call"])

        return json.dumps({k: str(v) for k, v in arg_dict.items()}, default=str)
    except Exception:
        return "Error capturing inputs"


@contextmanager
def trace_thread(session_id: str):
    if not session_id:
        yield
        return

    token = attach(baggage.set_baggage("gwenflow.session_id", str(session_id)))
    try:
        yield
    finally:
        detach(token)
