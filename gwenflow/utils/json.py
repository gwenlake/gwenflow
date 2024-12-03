import json

def parse_json_markdown(json_string: str) -> dict:
    # Remove the triple backticks if present
    json_string = json_string.strip()
    start_index = json_string.find("```json")
    end_index = json_string.find("```", start_index + len("```json"))

    if start_index != -1 and end_index != -1:
        extracted_content = json_string[start_index + len("```json"):end_index].strip()
        
        # Parse the JSON string into a Python dictionary
        parsed = json.loads(extracted_content)
    elif start_index != -1 and end_index == -1 and json_string.endswith("``"):
        end_index = json_string.find("``", start_index + len("```json"))
        extracted_content = json_string[start_index + len("```json"):end_index].strip()
        
        # Parse the JSON string into a Python dictionary
        parsed = json.loads(extracted_content)
    elif json_string.startswith("{"):
        # Parse the JSON string into a Python dictionary
        parsed = json.loads(json_string)
    else:
        raise Exception("Could not find JSON block in the output.")

    return parsed

def parse_and_check_json_markdown(text: str, expected_keys: list[str]) -> dict:
    try:
        json_obj = parse_json_markdown(text)
    except json.JSONDecodeError as e:
        raise Exception(f"Got invalid JSON object. Error: {e}")
    for key in expected_keys:
        if key not in json_obj:
            raise Exception(
                f"Got invalid return object. Expected key `{key}` "
                f"to be present, but got {json_obj}"
            )
    return json_obj

from datetime import date, datetime

def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""

    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError ("Type %s not serializable" % type(obj))

def to_json(obj):
    return json.dumps(obj, default=json_serial)
