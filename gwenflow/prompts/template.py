from __future__ import annotations

from pydantic import BaseModel, model_validator
from pathlib import Path
from typing import Any, Optional, Union
from string import Formatter


def _get_template_variables(template: str) -> list[str]:
    """Get the variables from the template.

    Args:
        template: The template string.
        template_format: The template format. Should be one of "f-string" or "jinja2".

    Returns:
        The variables from the template.

    Raises:
        ValueError: If the template format is not supported.
    """
    input_variables = {
        v for _, v, _, _ in Formatter().parse(template) if v is not None
    }

    return sorted(input_variables)


class PromptTemplate(BaseModel):
    """Prompt template for a language model."""

    template: str
    """The prompt template."""

    input_variables: list[str]
    """The prompt input variables."""
    
    @model_validator(mode="before")
    @classmethod
    def pre_init_validation(cls, values: dict) -> Any:
        """Check that template and input variables are consistent."""
        if values.get("template") is None:
            # pydantic will fail if template is not provided.
            return values
        values["input_variables"] = _get_template_variables(values["template"])
        return values

    def __str__(self) -> dict:
        return self.template

    def format(self, **kwargs: Any) -> str:
        """Format the prompt with the inputs.

        Args:
            kwargs: Any arguments to be passed to the prompt template.

        Returns:
            A formatted string.
        """
        return self.template.format(**kwargs)

    @classmethod
    def from_file(
        cls,
        template_file: Union[str, Path],
        encoding: Optional[str] = None,
    ) -> PromptTemplate:
        """Load a prompt from a file.

        Args:
            template_file: The path to the file containing the prompt template.
            encoding: The encoding system for opening the template file.
                If not provided, will use the OS default.

        Returns:
            The prompt loaded from the file.
        """
        with open(str(template_file), encoding=encoding) as f:
            template = f.read()
        return cls.from_template(template=template, role="user")

    @classmethod
    def from_template(
        cls,
        template: str,
        role: str = "user",
    ) -> PromptTemplate:
        """Load a prompt template from a template.

        Args:
            template: The template to load.
            kwargs: Any other arguments to pass to the prompt template.

        Returns:
            The prompt template loaded from the template.
        """
        return cls(template=template, role=role)