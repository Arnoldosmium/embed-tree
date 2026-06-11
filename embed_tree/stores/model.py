"""Legacy full-snapshot store contract."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

TreeState = dict[str, Any]


@runtime_checkable
class TreeStore(Protocol):
    """Load/save a complete internal ``EmbedTree`` snapshot.

    This is the current compatibility API used by ``EmbedTree(store=...)``.
    Newer loader/persister abstractions can be layered on top without changing
    this contract.
    """

    def load(self) -> TreeState | None:
        """Return the last saved state, or None if nothing persisted yet."""
        ...

    def save(self, state: TreeState) -> None:
        """Durably persist the given state, overwriting any previous one."""
        ...
