"""Abstract loader contract."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from embed_tree.representation import BranchNode


@runtime_checkable
class TreeLoader(Protocol):
    """Load a public tree representation from any source."""

    def load(self) -> BranchNode | None:
        """Return loaded tree data, or None if the source is empty."""
        ...
