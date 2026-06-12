"""Abstract reconciliation contract."""

from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable

from embed_tree.loaders import TreeLoader
from embed_tree.representation import BranchNode


@runtime_checkable
class TreeReconciler(Protocol):
    """Build a public tree from ground truth plus optional reusable state."""

    def reconcile(
        self,
        ground_truth_loader: TreeLoader,
        reusable_loader: TreeLoader | None = None,
        *,
        embedder: Callable[[Any], Any],
        config: Any | None = None,
    ) -> BranchNode | None:
        ...
