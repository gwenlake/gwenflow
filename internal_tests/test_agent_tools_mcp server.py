import asyncio
import json
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from typing import Dict, Any, List, Optional

app = FastAPI()

# Define tools that our MCP server will provide
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Add two numbers together",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"}
                },
                "required": ["a", "b"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"]
            }
        }
    }
]

# Implement the actual tool functionality
async def call_tool(tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    if tool_name == "add":
        return {"result": parameters["a"] + parameters["b"]}

    elif tool_name == "get_weather":
        # In a real implementation, you'd call an actual weather API
        weather_data = {
            "Tokyo": {"condition": "Sunny", "temperature": 25},
            "New York": {"condition": "Cloudy", "temperature": 18},
            "London": {"condition": "Rainy", "temperature": 15}
        }
        location = parameters["location"]
        if location in weather_data:
            return {"weather": weather_data[location]}
        return {"error": f"Weather data not available for {location}"}

    return {"error": f"Unknown tool: {tool_name}"}

async def sse_event_generator(request: Request):
    # Read the request body
    body_bytes = await request.body()
    body = json.loads(body_bytes)

    # Handle different MCP operations
    if body["action"] == "list_tools":
        yield f"data: {json.dumps({'tools': TOOLS})}\\\\n\\\\n"

    elif body["action"] == "call_tool":
        tool_name = body["tool_name"]
        parameters = body["parameters"]
        result = await call_tool(tool_name, parameters)
        yield f"data: {json.dumps({'result': result})}\\\\n\\\\n"

@app.post("/sse")
async def sse_endpoint(request: Request):
    return StreamingResponse(
        sse_event_generator(request),
        media_type="text/event-stream"
    )

@app.get("/sse")
async def sse_endpoint(request: Request):
    return StreamingResponse(
        sse_event_generator(request),
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)