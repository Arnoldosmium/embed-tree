"""Deterministic fake embeddings — no network, no extra deps.

Use for tests, demos, and offline development. The same text always maps to the
same vector (across processes), so it composes with persistence tests.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import EmbeddingProvider, _stable_seed


class FakeEmbeddingProvider(EmbeddingProvider):
    def __init__(self, dim: int = 32, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.dim = dim

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        out = np.empty((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            rng = np.random.default_rng(_stable_seed(t))
            out[i] = rng.normal(size=self.dim)
        return out
