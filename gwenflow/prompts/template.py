from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from string import Formatter
from typing import Optional, Union


def _get_template_variables(template: str) -> list[str]:
    input_variables = {v for _, v, _, _ in Formatter().parse(template) if v is not None}
    return sorted(input_variables)


@dataclass
class PromptTemplate:
    template: str
    input_variables: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.input_variables = _get_template_variables(self.template)

    def __str__(self) -> str:
        return self.template

    def format(self, **kwargs) -> str:
        return self.template.format(**kwargs)

    @classmethod
    def from_file(
        cls,
        template_file: Union[str, Path],
        encoding: Optional[str] = None,
    ) -> PromptTemplate:
        with open(str(template_file), encoding=encoding) as f:
            template = f.read()
        return cls(template=template)

    @classmethod
    def from_template(cls, template: str) -> PromptTemplate:
        return cls(template=template)
