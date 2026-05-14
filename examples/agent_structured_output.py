from typing import List

import dotenv
from pydantic import BaseModel

from gwenflow import Agent, ChatOpenAI

dotenv.load_dotenv(override=True)


class MovieReview(BaseModel):
    title: str
    year: int
    rating: float
    summary: str
    pros: List[str]
    cons: List[str]


agent = Agent(
    name="Movie Critic",
    instructions="You are a film critic. Analyse movies and return structured reviews.",
    llm=ChatOpenAI(model="gpt-4o-mini"),
    response_model=MovieReview,
)

movies = ["Inception (2010)", "The Matrix (1999)", "Interstellar (2014)"]

for movie in movies:
    response = agent.run(f"Review the movie: {movie}")
    review: MovieReview = response.parsed
    print(f"\n{review.title} ({review.year}) — {review.rating}/10")
    print(f"  {review.summary}")
    print(f"  Pros: {', '.join(review.pros)}")
    print(f"  Cons: {', '.join(review.cons)}")
