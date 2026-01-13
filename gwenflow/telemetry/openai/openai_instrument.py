from openinference.instrumentation.openai import OpenAIInstrumentor

class OpenAIInstrumentation:
    def __init__(self):
        self._instrumentor = OpenAIInstrumentor()

    def instrument(self):
        if not self._instrumentor.is_instrumented_by_opentelemetry:
            self._instrumentor.instrument()

openai_telemetry = OpenAIInstrumentation()