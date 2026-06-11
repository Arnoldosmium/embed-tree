"""Abstract reconciliation contract."""

from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable

from embed_tree.persisters.model import MaterializedTreeState
from embed_tree.loaders import TreeLoader
from embed_tree.representation import PartialTree


@runtime_checkable
class TreeReconciler(Protocol):
    """Build operational tree state from ground truth plus reusable state."""

    def reconcile(
        self,
        ground_truth_loader: TreeLoader,
        reusable_loader: TreeLoader | None = None,
        *,
        embedder: Callable[[Any], Any],
        config: Any | None = None,
    ) -> PartialTree | MaterializedTreeState:
        """Return reconciled representation or materialized tree state."""
        ...
