"""Abstract persister contract."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from embed_tree.representation import BranchNode

MaterializedTreeState = dict[str, Any]


@runtime_checkable
class TreePersister(Protocol):
    """Persist a tree representation or internal materialized state."""

    def save(self, state: BranchNode | MaterializedTreeState | Any) -> None:
        ...
