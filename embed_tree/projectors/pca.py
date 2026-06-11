"""PCA projector built on the existing reducer implementations."""

from __future__ import annotations

from typing import Any

import numpy as np

from embed_tree.config import TreeConfig
from embed_tree.reducers import Reducer

from .model import PCAConfig


class PCAProjector:
    """Callable PCA projection wrapper.

    This keeps the public model-facing API small while reusing the reducer
    state serialization and PCA implementations already used by EmbedTree.
    """

    def __init__(self, config: PCAConfig | None = None, *, reducer: Reducer | None = None) -> None:
        self.config = config or PCAConfig()
        self.reducer = reducer or Reducer.from_config(_tree_config(self.config))

    @property
    def is_fitted(self) -> bool:
        return self.reducer.is_fitted

    def fit(self, vectors: np.ndarray) -> None:
        self.reducer.fit(np.asarray(vectors, dtype=np.float64))

    def partial_fit(self, vectors: np.ndarray) -> None:
        self.reducer.partial_fit(np.asarray(vectors, dtype=np.float64))

    def transform(self, vectors: np.ndarray) -> np.ndarray:
        return self.reducer.transform(np.asarray(vectors, dtype=np.float64))

    def __call__(self, vectors: np.ndarray) -> np.ndarray:
        return self.transform(vectors)

    def to_dict(self) -> dict[str, Any]:
        return {"config": self.config.model_dump(), "reducer": self.reducer.to_dict()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PCAProjector":
        return cls(PCAConfig(**data["config"]), reducer=Reducer.from_dict(data["reducer"]))


def _tree_config(config: PCAConfig) -> TreeConfig:
    return TreeConfig(
        pca_dims=config.dims,
        pca_mode=config.mode,  # type: ignore[arg-type]
        pca_warmup=config.warmup,
        pca_batch_size=config.batch_size,
    )

