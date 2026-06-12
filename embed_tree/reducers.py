"""Dimensionality reducers (PCA) for embed-tree.

A reducer maps a raw embedding (dim D) to the lower-dim vector (dim d) used for
routing/clustering. It is JSON-serializable so the fitted projection survives a
restart without refitting.

Three kinds:
  - IdentityReducer        : passthrough; always "fitted" (pca_dims is None).
  - FreezePCAReducer       : fit once, frozen; fit() fully refits (rebalance).
  - IncrementalPCAReducer  : partial_fit() keeps updating the projection.

Transform is implemented directly from (components_, mean_) so it never needs
sklearn at query time:  reduced = (X - mean_) @ components_.T
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import TreeConfig


class Reducer:
    kind = "base"

    @property
    def is_fitted(self) -> bool:
        raise NotImplementedError

    def transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fit(self, X: np.ndarray) -> None:
        raise NotImplementedError

    def partial_fit(self, X: np.ndarray) -> None:
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    # --- factories ---------------------------------------------------------
    @staticmethod
    def from_config(cfg: TreeConfig) -> "Reducer":
        if cfg.pca_dims is None:
            return IdentityReducer()
        if cfg.pca_mode == "freeze":
            return FreezePCAReducer(cfg.pca_dims)
        return IncrementalPCAReducer(cfg.pca_dims)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Reducer":
        kind = d["kind"]
        if kind == "identity":
            return IdentityReducer()
        if kind == "pca_freeze":
            return FreezePCAReducer._load(d)
        if kind == "pca_incremental":
            return IncrementalPCAReducer._load(d)
        raise ValueError(f"unknown reducer kind: {kind!r}")


class IdentityReducer(Reducer):
    kind = "identity"

    @property
    def is_fitted(self) -> bool:
        return True

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float64)

    def fit(self, X: np.ndarray) -> None:
        pass

    def partial_fit(self, X: np.ndarray) -> None:
        pass

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind}


class _PCABase(Reducer):
    def __init__(self, n_components: int) -> None:
        self.n_components = n_components
        self.components_: np.ndarray | None = None  # (d, D)
        self.mean_: np.ndarray | None = None  # (D,)

    @property
    def is_fitted(self) -> bool:
        return self.components_ is not None

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.components_ is None or self.mean_ is None:
            raise RuntimeError("reducer is not fitted yet")
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) @ self.components_.T


class FreezePCAReducer(_PCABase):
    kind = "pca_freeze"

    def fit(self, X: np.ndarray) -> None:
        from sklearn.decomposition import PCA

        X = np.asarray(X, dtype=np.float64)
        k = min(self.n_components, X.shape[0], X.shape[1])
        pca = PCA(n_components=k).fit(X)
        self.components_ = np.asarray(pca.components_, dtype=np.float64)
        self.mean_ = np.asarray(pca.mean_, dtype=np.float64)

    def partial_fit(self, X: np.ndarray) -> None:
        # "freeze" has no online update; a refit is a full fit (used by rebalance).
        self.fit(X)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "n_components": self.n_components,
            "components_": None if self.components_ is None else self.components_.tolist(),
            "mean_": None if self.mean_ is None else self.mean_.tolist(),
        }

    @classmethod
    def _load(cls, d: dict[str, Any]) -> "FreezePCAReducer":
        r = cls(d["n_components"])
        if d.get("components_") is not None:
            r.components_ = np.asarray(d["components_"], dtype=np.float64)
            r.mean_ = np.asarray(d["mean_"], dtype=np.float64)
        return r


class IncrementalPCAReducer(_PCABase):
    kind = "pca_incremental"

    def __init__(self, n_components: int) -> None:
        super().__init__(n_components)
        self._ipca: Any | None = None  # sklearn IncrementalPCA, lazily created

    def _new_ipca(self) -> Any:
        from sklearn.decomposition import IncrementalPCA

        return IncrementalPCA(n_components=self.n_components)

    def _sync(self) -> None:
        self.components_ = np.asarray(self._ipca.components_, dtype=np.float64)
        self.mean_ = np.asarray(self._ipca.mean_, dtype=np.float64)

    def fit(self, X: np.ndarray) -> None:
        # (re)initialize the running estimator from scratch.
        self._ipca = self._new_ipca()
        self._ipca.partial_fit(np.asarray(X, dtype=np.float64))
        self._sync()

    def partial_fit(self, X: np.ndarray) -> None:
        if self._ipca is None:
            self._ipca = self._new_ipca()
        self._ipca.partial_fit(np.asarray(X, dtype=np.float64))
        self._sync()

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"kind": self.kind, "n_components": self.n_components}
        if self._ipca is not None and hasattr(self._ipca, "components_"):
            ip = self._ipca
            d["state"] = {
                "components_": ip.components_.tolist(),
                "mean_": ip.mean_.tolist(),
                "var_": ip.var_.tolist(),
                "singular_values_": ip.singular_values_.tolist(),
                "n_samples_seen_": int(ip.n_samples_seen_),
                "n_components_": int(ip.n_components_),
                "n_features_in_": int(ip.n_features_in_),
            }
        return d

    @classmethod
    def _load(cls, d: dict[str, Any]) -> "IncrementalPCAReducer":
        r = cls(d["n_components"])
        state = d.get("state")
        if state is not None:
            ip = r._new_ipca()
            # restore just enough state to resume partial_fit and to transform
            ip.components_ = np.asarray(state["components_"], dtype=np.float64)
            ip.mean_ = np.asarray(state["mean_"], dtype=np.float64)
            ip.var_ = np.asarray(state["var_"], dtype=np.float64)
            ip.singular_values_ = np.asarray(state["singular_values_"], dtype=np.float64)
            ip.n_samples_seen_ = state["n_samples_seen_"]
            ip.n_components_ = state["n_components_"]
            ip.n_features_in_ = state["n_features_in_"]
            r._ipca = ip
            r._sync()
        return r
