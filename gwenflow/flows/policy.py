import fnmatch
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gwenflow.flows.flows import FlowNodeError, Node, resolve_credentials
from gwenflow.flows.handlers import node_paths


class FlowPolicyError(FlowNodeError):
    pass


@dataclass(frozen=True)
class FlowPolicy:
    node_types: Collection[str] | None = None
    credentials: Collection[str] | None = None
    root: str | Path | None = None

    def __post_init__(self) -> None:
        for name in ("node_types", "credentials"):
            value = getattr(self, name)
            if isinstance(value, str):
                object.__setattr__(self, name, (value,))

    def resolve(self, node: Node) -> dict[str, Any]:
        violations = []
        if self.node_types is not None and node.type not in self.node_types:
            violations.append(f"node type '{node.type}' is not allowed")
        if self.credentials is not None:
            for param, variable in node.credentials.items():
                if not any(fnmatch.fnmatchcase(variable, pattern) for pattern in self.credentials):
                    violations.append(f"credential '{param}' reads {variable}, which is not allowed")
        if violations:
            raise FlowPolicyError(node.name, "; ".join(violations))
        parameters = resolve_credentials(node)
        if self.root is not None:
            root = Path(self.root).resolve()
            outside = [p for p in node_paths(node.type, parameters) if not Path(p).resolve().is_relative_to(root)]
            if outside:
                raise FlowPolicyError(node.name, f"path outside {root}: {', '.join(outside)}")
        return parameters
