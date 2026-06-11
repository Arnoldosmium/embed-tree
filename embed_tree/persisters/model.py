"""Abstract persister contract."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from embed_tree.representation import PartialTree

MaterializedTreeState = dict[str, Any]


@runtime_checkable
class TreePersister(Protocol):
    """Persist a tree representation.

    The same protocol covers durable internal state, reusable state for future
    builds, and user-facing exports. The role comes from where the persister is
    passed, not from a separate cache type.
    """

    def save(self, state: PartialTree | MaterializedTreeState | Any) -> None:
        """Persist a tree representation or export artifact."""
        ...
