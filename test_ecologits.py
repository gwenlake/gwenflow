from ecologits import EcoLogits
from gwenflow.llms.openai.chat import ChatOpenAI
from dotenv import load_dotenv
from gwenflow.telemetry.base import TelemetryBase
load_dotenv()
import os


print(os.environ.get("TELEMETRY_ENDPOINT"))
EcoLogits.init(providers=["openai"])
telemetry = TelemetryBase(service_name="gwenflow-dev")
telemetry.initialize()
client = ChatOpenAI(model="gpt-5")

response = client.invoke(
    input=[
        {"role": "user", "content": "Tell me a funny joke!"}
    ]
)
print(response)