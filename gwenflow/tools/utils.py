import inspect


def function_to_json(func, name: str = None, description: str = None) -> dict:
    """Converts a Python function into a JSON-serializable dictionary that describes the function's signature, including its name, description, and parameters.

    Args:
        func: The function to be converted.
        name: Optional; The name to use for the function in the JSON output. If not provided, the function's __name__ attribute is used.
        description: Optional; A description of the function. If not provided, the function's docstring is used.

    Returns:
        A dictionary representing the function's signature in JSON format.
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

    parameters = {}
    for param in signature.parameters.values():
        try:
            param_type = type_map.get(param.annotation, "string")
        except KeyError as e:
            raise KeyError(f"Unknown type annotation {param.annotation} for parameter {param.name}: {str(e)}") from e

        if hasattr(param.default, "description"):
            parameters[param.name] = {
                "description": param.default.description,
                "type": param_type,
            }
        else:
            parameters[param.name] = {"type": param_type}

    required = [param.name for param in signature.parameters.values() if param.default == inspect._empty]

    return {
        "type": "function",
        "function": {
            "name": name or func.__name__,
            "description": description or func.__doc__ or "",
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required,
            },
        },
    }
