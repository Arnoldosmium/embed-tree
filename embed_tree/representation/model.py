"""Public tree representation models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Hashable, Sequence, TypeAlias

NodeId: TypeAlias = Hashable
VectorData: TypeAlias = Sequence[float]


@dataclass
class ContentNode:
    """Leaf content that can be embedded and inserted into an EmbedTree."""

    id: NodeId
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: VectorData | None = None


@dataclass
class BranchNode:
    """Recursive public tree branch."""

    id: NodeId
    label: str | None = None
    children: list["BranchNode | ContentNode"] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    vector_sum: VectorData | None = None
    _count: int | None = None

    @property
    def count(self) -> int:
        if self._count is not None:
            return self._count
        total = 0
        for child in self.children:
            total += child.count if isinstance(child, BranchNode) else 1
        return total

    @count.setter
    def count(self, value: int | None) -> None:
        self._count = value
