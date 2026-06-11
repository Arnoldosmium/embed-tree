"""OpenAI embedding adapter.

Requires the optional `openai` extra:  pip install "embed-tree[openai]"

Pass your (enterprise) key explicitly via `api_key`, or inject a pre-built
`client`. If both are omitted the OpenAI SDK falls back to its own credential
resolution.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import EmbeddingProvider


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        *,
        api_key: str | None = None,
        dimensions: int | None = None,  # shorten output (text-embedding-3-*)
        client: Any | None = None,  # inject for testing / custom config
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as e:  # pragma: no cover
                raise ImportError(
                    'OpenAIEmbeddingProvider needs the "openai" extra: '
                    'pip install "embed-tree[openai]"'
                ) from e
            client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self.client = client
        self.model = model
        self.dimensions = dimensions

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        kwargs: dict[str, Any] = {"model": self.model, "input": texts}
        if self.dimensions is not None:
            kwargs["dimensions"] = self.dimensions
        resp = self.client.embeddings.create(**kwargs)
        # OpenAI returns embeddings in input order.
        return np.asarray([d.embedding for d in resp.data], dtype=np.float32)
