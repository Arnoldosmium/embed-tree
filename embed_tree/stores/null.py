"""In-memory/no-op full-snapshot store."""

from __future__ import annotations

from .model import TreeState


class NullTreeStore:
    """No-op store for tests / pure in-memory use."""

    def load(self) -> TreeState | None:
        return None

    def save(self, state: TreeState) -> None:
        pass

