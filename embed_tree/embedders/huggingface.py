"""Open-source local embeddings via Hugging Face sentence-transformers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from embed_tree.providers.local import SentenceTransformerProvider


class HuggingFaceTextEmbedder(SentenceTransformerProvider):
    """Sentence-transformers embedder with Mac-friendly device selection.

    The model is downloaded from Hugging Face by sentence-transformers on first
    use and then cached by that stack. On Apple Silicon, device="auto" prefers
    MPS when PyTorch reports it as available; otherwise it falls back to CPU.
    """

    def __init__(
        self,
        model: str = "BAAI/bge-small-en-v1.5",
        *,
        device: str | None = "auto",
        cache_folder: str | Path | None = None,
        model_obj: Any | None = None,
        encode_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        resolved_device = _resolve_device(device)
        self.model_name = model
        self.device = resolved_device
        self.cache_folder = None if cache_folder is None else str(cache_folder)
        if model_obj is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:  # pragma: no cover
                raise ImportError(
                    'HuggingFaceTextEmbedder needs the "local" extra: '
                    'pip install "embed-tree[local]"'
                ) from e
            model_obj = SentenceTransformer(model, device=resolved_device, cache_folder=self.cache_folder)
            super().__init__(model=model, device=resolved_device, model_obj=model_obj, encode_kwargs=encode_kwargs, **kwargs)
        else:
            super().__init__(model=model, device=resolved_device, model_obj=model_obj, encode_kwargs=encode_kwargs, **kwargs)


def _resolve_device(device: str | None) -> str | None:
    if device != "auto":
        return device
    try:
        import torch
    except ImportError:
        return None
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def embed_texts(embedder: Any, texts: list[str]) -> np.ndarray:
    """Embed a batch through any callable or TextEmbedder-like object."""
    batch_fn = getattr(embedder, "embed_batch", None)
    if callable(batch_fn):
        return np.asarray(batch_fn(texts))
    return np.asarray([embedder(text) for text in texts])

