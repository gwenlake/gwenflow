import dotenv
from gwenflow import ChatOpenAI, Agent, Task

# Load API key from .env file
dotenv.load_dotenv(override=True)

# Define the language model
llm = ChatOpenAI(model="gpt-4o-mini")

# Define the sentiment analysis agent
sentiment_agent = Agent(
    role="Sentiment Analyst",
    instructions="Analyze the sentiment of the provided text and determine whether it is positive, negative, or neutral.",
    llm=llm
)

# Define the sentiment analysis task
def analyze_sentiment(text: str) -> str:
    """Analyze the sentiment of the given text."""
    task = Task(
        description=f"Analyze the sentiment of the following text: '{text}'",
        expected_output="Indicate whether the sentiment is positive, negative, or neutral.",
        agent=sentiment_agent
    )
    return task.run()

# Example usage
text = "I am very happy with the quality of the service."
result = analyze_sentiment(text)
print(f"Text: {text}\nSentiment: {result}")