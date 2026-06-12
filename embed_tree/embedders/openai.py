"""OpenAI text embeddings."""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseTextEmbedder


class OpenAITextEmbedder(BaseTextEmbedder):
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        *,
        api_key: str | None = None,
        dimensions: int | None = None,
        client: Any | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as e:  # pragma: no cover
                raise ImportError(
                    'OpenAITextEmbedder needs the "openai" extra: '
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
        return np.asarray([d.embedding for d in resp.data], dtype=np.float32)
