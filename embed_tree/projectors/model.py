"""Vector projection contracts."""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel


class PCAConfig(BaseModel):
    """Configuration for PCA-based projection."""

    dims: int | None = None
    mode: Literal["freeze", "incremental"] = "freeze"
    warmup: int = 1000
    batch_size: int = 256


@runtime_checkable
class VectorProjector(Protocol):
    """Map raw vectors to operational vectors."""

    @property
    def is_fitted(self) -> bool:
        """Whether this projector can transform vectors."""
        ...

    def fit(self, vectors: np.ndarray) -> None:
        """Fit projector state from vectors."""
        ...

    def transform(self, vectors: np.ndarray) -> np.ndarray:
        """Project vectors."""
        ...

    def __call__(self, vectors: np.ndarray) -> np.ndarray:
        """Project vectors."""
        ...
