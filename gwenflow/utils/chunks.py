
def merge_fields(target, source):
    for key, value in source.items():
        if isinstance(value, str):
            target[key] += value
        elif value is not None and isinstance(value, dict):
            merge_fields(target[key], value)


def merge_chunk(final_response: dict, delta: dict) -> None:
    delta.pop("role", None)
    merge_fields(final_response, delta)

    tool_calls = delta.get("tool_calls")
    if tool_calls and len(tool_calls) > 0:
        if tool_calls[0].get("index"):
            index = tool_calls[0].pop("index")
        else:
            index = 0
        merge_fields(final_response["tool_calls"][index], tool_calls[0])
