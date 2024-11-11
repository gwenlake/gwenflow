from __future__ import annotations

from pydantic import BaseModel, model_validator
from typing import Any, Optional, Union

from gwenflow.prompts.template import PromptTemplate


class PipelinePromptTemplate(BaseModel):
    """Pipeline prompt template for a language model."""

    prompts: list[PromptTemplate]
    """The pipeline."""

    input_variables: list[str]
    """The prompt input variables."""
    
    @model_validator(mode="before")
    @classmethod
    def get_input_variables(cls, values: dict) -> Any:
        """Get input variables."""
        all_variables = set()
        for prompt in values["prompts"]:
            all_variables.update(prompt.input_variables)
        values["input_variables"] = list(all_variables)
        return values


    def format(self, **kwargs: Any) -> list:
        """Format the prompt with the inputs.

        Args:
            kwargs: Any arguments to be passed to the prompt template.

        Returns:
            A formatted string.
        """
        _formatted = []
        for prompt in self.prompts:
            _formatted.append( prompt.format(**kwargs) )
        return _formatted