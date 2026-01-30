from pydantic import BaseModel, Field


class UsageInputDetails(BaseModel):
    cached_tokens: int = 0

    def add(self, other: "UsageInputDetails") -> None:
        self.cached_tokens += other.cached_tokens


class UsageReasoning(BaseModel):
    reasoning_tokens: int = 0

    def add(self, other: "UsageReasoning") -> None:
        self.reasoning_tokens += other.reasoning_tokens


class Usage(BaseModel):
    requests: int = 0
    input_tokens: int = 0
    input_tokens_details: UsageInputDetails = Field(default_factory=UsageInputDetails)
    output_tokens: int = 0
    output_tokens_details: UsageReasoning = Field(default_factory=UsageReasoning)
    total_tokens: int = 0

    def add(self, other: "Usage") -> None:
        self.requests += other.requests if other.requests else 0
        self.input_tokens += other.input_tokens if other.input_tokens else 0
        self.output_tokens += other.output_tokens if other.output_tokens else 0
        self.total_tokens += other.total_tokens if other.total_tokens else 0

        if other.input_tokens_details:
            self.input_tokens_details.add(other.input_tokens_details)
        if other.output_tokens_details:
            self.output_tokens_details.add(other.output_tokens_details)
