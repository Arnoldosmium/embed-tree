"""Default tree reconciler."""

from __future__ import annotations

from typing import Any, Callable

from embed_tree.loaders import TreeLoader
from embed_tree.representation import BranchNode


class DefaultTreeReconciler:
    """Return the current ground-truth tree.

    Reusable embedding/cache reconciliation is intentionally not part of the
    public 0.1 tree model. Embedding caches live on ContentNode.embedding.
    """

    def reconcile(
        self,
        ground_truth_loader: TreeLoader,
        reusable_loader: TreeLoader | None = None,
        *,
        embedder: Callable[[Any], Any],
        config: Any | None = None,
    ) -> BranchNode | None:
        return ground_truth_loader.load()
