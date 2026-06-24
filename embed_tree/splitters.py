"""Split strategies for building embed-tree hierarchies."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Protocol

import numpy as np
from sklearn.cluster import KMeans

from .config import TreeConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SplitDecision:
    """A splitter's recommendation for one group of content items."""

    should_split: bool
    clusters: list[list[Any]]
    score: float | None = None
    reason: str | None = None


class SplitStrategy(Protocol):
    """Decide whether and how to split a group of stored content items."""

    def split(self, items: list[Any], *, depth: int = 0) -> SplitDecision:
        ...


class FixedKMeansSplitter:
    """Capacity-driven KMeans splitter that preserves the original behavior."""

    def __init__(self, config: TreeConfig) -> None:
        self.config = config

    def split(self, items: list[Any], *, depth: int = 0) -> SplitDecision:
        if len(items) <= self.config.leaf_target:
            _log(
                self.config,
                "split.fixed.skip",
                n=len(items),
                depth=depth,
                reason="within_leaf_target",
                leaf_target=self.config.leaf_target,
            )
            return SplitDecision(False, [], reason="within_leaf_target")

        buckets = _kmeans_buckets(items, min(self.config.max_branches, len(items)), self.config)
        if len(buckets) < 2:
            _log(self.config, "split.fixed.skip", n=len(items), depth=depth, reason="unsplittable")
            return SplitDecision(False, [], reason="unsplittable")
        _log(
            self.config,
            "split.fixed.accept",
            n=len(items),
            depth=depth,
            k=len(buckets),
            sizes=_sizes(buckets.values()),
        )
        return SplitDecision(True, list(buckets.values()))


class AdaptiveKMeansSplitter:
    """Discover a useful KMeans split instead of forcing a fixed fan-out."""

    def __init__(self, config: TreeConfig) -> None:
        self.config = config

    def split(self, items: list[Any], *, depth: int = 0) -> SplitDecision:
        if len(items) <= self.config.leaf_target:
            _log(
                self.config,
                "split.adaptive.skip",
                n=len(items),
                depth=depth,
                reason="within_leaf_target",
                leaf_target=self.config.leaf_target,
            )
            return SplitDecision(False, [], reason="within_leaf_target")

        vectors = _vectors(items)
        parent_centroid = vectors.mean(axis=0)
        parent_dispersion = _mean_distance(vectors, parent_centroid)
        min_parent_dispersion = _effective_min_parent_dispersion(self.config, depth)
        if parent_dispersion < min_parent_dispersion:
            _log(
                self.config,
                "split.adaptive.skip",
                n=len(items),
                depth=depth,
                reason="coherent",
                parent_dispersion=parent_dispersion,
                min_parent_dispersion=min_parent_dispersion,
                base_min_parent_dispersion=self.config.min_parent_dispersion,
                parent_dispersion_decay=self.config.parent_dispersion_decay,
            )
            return SplitDecision(False, [], score=parent_dispersion, reason="coherent")

        max_k = min(self.config.max_branches, len(items) // self.config.min_cluster_size, _distinct_vector_count(vectors))
        if max_k < 2:
            _log(
                self.config,
                "split.adaptive.skip",
                n=len(items),
                depth=depth,
                reason="too_few_supported_clusters",
                parent_dispersion=parent_dispersion,
                max_k=max_k,
            )
            return SplitDecision(False, [], score=parent_dispersion, reason="too_few_supported_clusters")

        best: _Candidate | None = None
        for k in range(2, max_k + 1):
            buckets = _kmeans_buckets(items, k, self.config)
            if len(buckets) < 2:
                _log(
                    self.config,
                    "split.adaptive.reject",
                    n=len(items),
                    depth=depth,
                    k=k,
                    reason="single_cluster",
                    parent_dispersion=parent_dispersion,
                )
                continue
            if any(len(bucket) < self.config.min_cluster_size for bucket in buckets.values()):
                _log(
                    self.config,
                    "split.adaptive.reject",
                    n=len(items),
                    depth=depth,
                    k=k,
                    reason="small_cluster",
                    sizes=_sizes(buckets.values()),
                    min_cluster_size=self.config.min_cluster_size,
                    parent_dispersion=parent_dispersion,
                )
                continue

            candidate = _score_candidate(k, list(buckets.values()), parent_dispersion, self.config)
            _log(
                self.config,
                "split.adaptive.candidate",
                n=len(items),
                depth=depth,
                k=k,
                sizes=candidate.sizes,
                parent_dispersion=parent_dispersion,
                child_dispersion=candidate.child_dispersion,
                gain=candidate.gain,
                gain_ratio=candidate.gain_ratio,
                separation=candidate.separation,
                imbalance=candidate.imbalance,
                score=candidate.score,
            )
            if best is None or candidate.score > best.score:
                best = candidate

        if best is None:
            _log(
                self.config,
                "split.adaptive.skip",
                n=len(items),
                depth=depth,
                reason="no_valid_split",
                parent_dispersion=parent_dispersion,
            )
            return SplitDecision(False, [], score=parent_dispersion, reason="no_valid_split")

        if not _passes_gain_threshold(best, self.config):
            _log(
                self.config,
                "split.adaptive.skip",
                n=len(items),
                depth=depth,
                reason="low_gain",
                best_k=best.k,
                sizes=best.sizes,
                gain=best.gain,
                gain_ratio=best.gain_ratio,
                min_split_gain=self.config.min_split_gain,
                min_split_gain_ratio=self.config.min_split_gain_ratio,
                score=best.score,
            )
            return SplitDecision(False, [], score=best.score, reason="low_gain")

        _log(
            self.config,
            "split.adaptive.accept",
            n=len(items),
            depth=depth,
            k=best.k,
            sizes=best.sizes,
            gain=best.gain,
            gain_ratio=best.gain_ratio,
            score=best.score,
        )
        return SplitDecision(True, best.clusters, score=best.score)


def make_splitter(config: TreeConfig) -> SplitStrategy:
    if config.split_mode == "adaptive":
        return AdaptiveKMeansSplitter(config)
    return FixedKMeansSplitter(config)


@dataclass(frozen=True)
class _Candidate:
    k: int
    clusters: list[list[Any]]
    sizes: list[int]
    score: float
    gain: float
    gain_ratio: float
    child_dispersion: float
    separation: float
    imbalance: float


def _score_candidate(k: int, clusters: list[list[Any]], parent_dispersion: float, config: TreeConfig) -> _Candidate:
    total = sum(len(cluster) for cluster in clusters)
    weighted_child_dispersion = 0.0
    centroids: list[np.ndarray] = []
    sizes: list[int] = []

    for cluster in clusters:
        vectors = _vectors(cluster)
        centroid = vectors.mean(axis=0)
        centroids.append(centroid)
        sizes.append(len(cluster))
        weighted_child_dispersion += (len(cluster) / total) * _mean_distance(vectors, centroid)

    gain = parent_dispersion - weighted_child_dispersion
    gain_ratio = gain / parent_dispersion if parent_dispersion > 0 else 0.0
    separation = _mean_pairwise_distance(centroids)
    imbalance = _imbalance(sizes)
    score = gain + config.separation_weight * separation - config.imbalance_weight * imbalance
    return _Candidate(
        k=k,
        clusters=clusters,
        sizes=sizes,
        score=score,
        gain=gain,
        gain_ratio=gain_ratio,
        child_dispersion=weighted_child_dispersion,
        separation=separation,
        imbalance=imbalance,
    )


def _effective_min_parent_dispersion(config: TreeConfig, depth: int) -> float:
    return config.min_parent_dispersion * (config.parent_dispersion_decay**depth)


def _passes_gain_threshold(candidate: _Candidate, config: TreeConfig) -> bool:
    if candidate.gain >= config.min_split_gain:
        return True
    return config.min_split_gain_ratio is not None and candidate.gain_ratio >= config.min_split_gain_ratio


def _kmeans_buckets(items: list[Any], k: int, config: TreeConfig) -> dict[int, list[Any]]:
    vectors = _vectors(items)
    effective_k = min(k, _distinct_vector_count(vectors))
    if effective_k < 2:
        return {0: items}

    labels = KMeans(n_clusters=effective_k, n_init="auto", **config.model_args).fit_predict(vectors)
    buckets: dict[int, list[Any]] = {}
    for label, item in zip(labels, items):
        buckets.setdefault(int(label), []).append(item)
    return buckets


def _vectors(items: list[Any]) -> np.ndarray:
    return np.stack([item.vector for item in items])


def _distinct_vector_count(vectors: np.ndarray) -> int:
    return int(np.unique(vectors, axis=0).shape[0])


def _mean_distance(vectors: np.ndarray, centroid: np.ndarray) -> float:
    if len(vectors) == 0:
        return 0.0
    return float(np.linalg.norm(vectors - centroid, axis=1).mean())


def _mean_pairwise_distance(centroids: list[np.ndarray]) -> float:
    if len(centroids) < 2:
        return 0.0
    distances: list[float] = []
    for i, left in enumerate(centroids[:-1]):
        for right in centroids[i + 1 :]:
            distances.append(float(np.linalg.norm(left - right)))
    return float(np.mean(distances)) if distances else 0.0


def _imbalance(sizes: list[int]) -> float:
    if not sizes:
        return 0.0
    mean = float(np.mean(sizes))
    return float(np.std(sizes) / mean) if mean > 0 else 0.0


def _sizes(clusters: Any) -> list[int]:
    return [len(cluster) for cluster in clusters]


def _log(config: TreeConfig, event: str, **fields: Any) -> None:
    if not config.log_split_decisions:
        return
    formatted = " ".join(f"{key}={_format_value(value)}" for key, value in fields.items())
    logger.info("%s %s", event, formatted)


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, list):
        return "[" + ",".join(_format_value(item) for item in value) + "]"
    return str(value)
