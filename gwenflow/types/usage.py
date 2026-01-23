from pydantic import BaseModel, Field


class UsageDetails(BaseModel):
    cached_tokens: int = 0
    reasoning_tokens: int = 0

    def add(self, other: "UsageDetails") -> None:
        self.cached_tokens += other.cached_tokens
        self.reasoning_tokens += other.reasoning_tokens

class Usage(BaseModel):
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    input_tokens_details: UsageDetails = Field(default_factory=UsageDetails)
    output_tokens_details: UsageDetails = Field(default_factory=UsageDetails)

    def add(self, other: "Usage") -> None:
        self.requests += other.requests if other.requests else 0
        self.input_tokens += other.input_tokens if other.input_tokens else 0
        self.output_tokens += other.output_tokens if other.output_tokens else 0
        self.total_tokens += other.total_tokens if other.total_tokens else 0

        if other.input_tokens_details:
            self.input_tokens_details.add(other.input_tokens_details)
        if other.output_tokens_details:
            self.output_tokens_details.add(other.output_tokens_details)
