"""Embedding model contracts."""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

import numpy as np

Vector = np.ndarray


@runtime_checkable
class TextEmbedder(Protocol):
    """Turn strings into embedding vectors."""

    def embed(self, text: str) -> Vector:
        """Embed one string."""
        ...

    def embed_batch(self, texts: Sequence[str]) -> np.ndarray:
        """Embed strings in order."""
        ...

    def __call__(self, text: str) -> Vector:
        """Embed one string."""
        ...

