"""Abstract loader contract."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from embed_tree.representation import PartialTree


@runtime_checkable
class TreeLoader(Protocol):
    """Load a partial tree from any source.

    Ground truth and reusable-state inputs share this shape. Their semantics
    come from the argument position where the loader is used.
    """

    def load(self) -> PartialTree | None:
        """Return loaded tree data, or None if the source is empty."""
        ...
