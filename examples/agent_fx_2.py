import json

import dotenv
import requests

from gwenflow import Agent, ChatOpenAI, FunctionTool

dotenv.load_dotenv(override=True)


def get_bike_availability(station_name: str = None) -> str:
    """Get the current number of bikes available at a given Rennes station. The station_name must match exactly the station name in the API. Otherwise, we return the full record. Returns JSON string with station data or error message."""
    url_station = f"https://data.rennesmetropole.fr/api/explore/v2.1/catalog/datasets/etat-des-stations-le-velo-star-en-temps-reel/records?where=nom=%22{station_name}%22&limit=1"
    url_general = "https://data.rennesmetropole.fr/api/explore/v2.1/catalog/datasets/etat-des-stations-le-velo-star-en-temps-reel/records"

    try:
        if not station_name:
            response = requests.get(url_general)
        else:
            response = requests.get(url_station)
        response.raise_for_status()
        data = response.json()
        records = data.get("results", [])
        if not records:
            return json.dumps({"error": "Station not found"})
        return json.dumps(records)
    except Exception as e:
        print(f"Error retrieving data for station '{station_name}': {e}")
        return json.dumps({"error": "Failed to retrieve data"})


tool_get_bike_availability = FunctionTool.from_function(get_bike_availability)


llm = ChatOpenAI(model="gpt-4o-mini")

agent = Agent(
    name="Agent City Data",
    instructions=["Get recent data on bike traffic in the city", "Answer in one sentence", "Always use your tools"],
    llm=llm,
    tools=[tool_get_bike_availability],
)

queries = [
    "How many bike are available at the Oberthur station ?",
    "How many bikes spot are in Sainte-Anne station ?",
    "How many bike docking stations are there in total ?",
    "At which station should I head for a bike if I'm located at 48.115501, -1.665948",
]

for query in queries:
    print("")
    print("Q:", query)
    print("A:", agent.run(query).content)
