"""Deprecated cache compatibility contract."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from embed_tree.loaders.model import TreeLoader
from embed_tree.persisters.model import MaterializedTreeState, TreePersister


@runtime_checkable
class TreeCache(TreeLoader, TreePersister, Protocol):
    """Deprecated alias for TreeLoader + TreePersister."""
