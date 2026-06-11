"""Shared embedding-provider machinery.

A provider turns content (text) into a vector. Concrete providers only need to
implement `_embed_batch`; this base adds the engineering wrapper every real
provider needs (see DESIGN.md note on §2):

  - **batching**: callers can embed many texts in one backend call;
  - **caching**: identical content is never re-embedded (or re-billed);
  - **retry + backoff**: API calls inevitably 429/timeout;
  - **optional normalize**: off by default — the tree already L2-normalizes for
    cosine, so leave it off unless you use the provider standalone.

A provider is callable, so it drops straight into `EmbedTree(embedder=provider)`.
"""

from __future__ import annotations

import hashlib
import random
import time
from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np

Vector = np.ndarray


class EmbeddingProvider(ABC):
    def __init__(
        self,
        *,
        cache: bool = True,
        normalize: bool = False,
        max_retries: int = 5,
        backoff_base: float = 0.5,
    ) -> None:
        self._cache: dict[str, Vector] | None = {} if cache else None
        self.normalize = normalize
        self.max_retries = max_retries
        self.backoff_base = backoff_base

    # --- implement this in subclasses (no caching/retry needed here) -------
    @abstractmethod
    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed texts in order; return array of shape (len(texts), dim)."""

    # --- public API --------------------------------------------------------
    def embed_batch(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        out: list[Vector | None] = [None] * len(texts)
        miss_idx: list[int] = []
        miss_txt: list[str] = []
        for i, t in enumerate(texts):
            cached = self._cache.get(t) if self._cache is not None else None
            if cached is not None:
                out[i] = cached
            else:
                miss_idx.append(i)
                miss_txt.append(t)

        if miss_txt:
            vecs = self._post(self._with_retry(miss_txt))
            for j, i in enumerate(miss_idx):
                out[i] = vecs[j]
                if self._cache is not None:
                    self._cache[texts[i]] = vecs[j]

        return np.stack(out)  # type: ignore[arg-type]

    def embed(self, text: str) -> Vector:
        return self.embed_batch([text])[0]

    def __call__(self, content: str) -> Vector:
        return self.embed(content)

    def cache_clear(self) -> None:
        if self._cache is not None:
            self._cache.clear()

    # --- internals ---------------------------------------------------------
    def _post(self, vecs: np.ndarray) -> np.ndarray:
        vecs = np.asarray(vecs, dtype=np.float32)
        if self.normalize:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vecs = vecs / norms
        return vecs

    def _with_retry(self, texts: list[str]) -> np.ndarray:
        for attempt in range(self.max_retries + 1):
            try:
                return np.asarray(self._embed_batch(texts))
            except Exception:
                if attempt >= self.max_retries:
                    raise
                delay = self.backoff_base * (2**attempt) + random.uniform(0, 0.1)
                time.sleep(delay)
        raise AssertionError("unreachable")  # pragma: no cover


def _stable_seed(text: str) -> int:
    """Deterministic 32-bit seed from text (process-independent)."""
    return int.from_bytes(hashlib.md5(text.encode("utf-8")).digest()[:4], "big")
