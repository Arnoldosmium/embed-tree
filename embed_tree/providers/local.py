"""Local (self-hosted) embeddings via sentence-transformers.

Requires the optional `local` extra:  pip install "embed-tree[local]"

The model is downloaded from Hugging Face on first use and runs locally — no
data leaves the machine. Pick a model from the MTEB leaderboard; the default is
a small, fast, solid general-purpose English model.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import EmbeddingProvider


class SentenceTransformerProvider(EmbeddingProvider):
    def __init__(
        self,
        model: str = "BAAI/bge-small-en-v1.5",
        *,
        device: str | None = None,  # "cpu" | "cuda" | "mps" | None=auto
        model_obj: Any | None = None,  # inject a preloaded SentenceTransformer
        encode_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if model_obj is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:  # pragma: no cover
                raise ImportError(
                    'SentenceTransformerProvider needs the "local" extra: '
                    'pip install "embed-tree[local]"'
                ) from e
            model_obj = SentenceTransformer(model, device=device)
        self.model = model_obj
        self.encode_kwargs = encode_kwargs or {}

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(texts, convert_to_numpy=True, **self.encode_kwargs)
        return np.asarray(vecs, dtype=np.float32)
