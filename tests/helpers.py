"""Test-only embedders."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from embed_tree import BaseTextEmbedder


class FakeTextEmbedder(BaseTextEmbedder):
    def __init__(self, dim: int = 32, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.dim = dim

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        out = np.empty((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            seed = int.from_bytes(hashlib.md5(text.encode("utf-8")).digest()[:4], "big")
            rng = np.random.default_rng(seed)
            out[i] = rng.normal(size=self.dim)
        return out
