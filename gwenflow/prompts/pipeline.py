from __future__ import annotations

from dataclasses import dataclass, field

from gwenflow.memory.prompts.template import PromptTemplate


@dataclass
class PipelinePromptTemplate:
    prompts: list[PromptTemplate] = field(default_factory=list)
    input_variables: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        all_variables = set()
        for prompt in self.prompts:
            all_variables.update(prompt.input_variables)
        self.input_variables = list(all_variables)

    def format(self, **kwargs) -> list:
        return [prompt.format(**kwargs) for prompt in self.prompts]
