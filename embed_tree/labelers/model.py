"""Labeling model contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Protocol, runtime_checkable


@dataclass(frozen=True)
class LabelCandidate:
    """Nearby node or item used as context for a label."""

    id: Any
    text: str
    distance: float | None = None
    payload: Any = None


@dataclass(frozen=True)
class LabelRequest:
    """Context for generating a label."""

    candidates: list[LabelCandidate]
    max_words: int = 6


@runtime_checkable
class Labeler(Protocol):
    """Generate a label from nearby candidates."""

    def stream(self, request: LabelRequest) -> Iterable[str]:
        """Yield label chunks."""
        ...

    def label(self, request: LabelRequest) -> str:
        """Return the full label."""
        ...

