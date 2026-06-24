"""Configuration for embed-tree, as a single pydantic object.

`TreeConfig` is a plain pydantic `BaseModel` (NOT `BaseSettings`): it is
constructed explicitly and handed whole to `EmbedTree(config=...)`. It does
**not** read environment variables — every value must be passed in code, so the
configuration is always explicit and reproducible.

PCA dimensionality reduction can run in freeze or incremental mode; see the
reducer implementations for the rebalance contract.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class RebalanceConfig(BaseModel):
    """When/whether to rebuild the whole tree from its leaves."""

    enabled: bool = True
    every_n_inserts: int | None = 10_000  # auto-rebuild cadence; None disables
    on_demand: bool = True  # allow manual tree.rebalance()


class LLMConfig(BaseModel):
    """How to auto-name taxonomy nodes. Provider "none" uses a no-network
    keyword labeler; "openai" and "local" generate labels with an LLM.
    """

    provider: Literal["none", "openai", "local"] = "none"
    model: str = "gpt-4o-mini"  # OpenAI model id, or HF model id when local
    api_key: str | None = None  # OpenAI key (explicit; no env)
    base_url: str | None = None  # OpenAI-compatible endpoint (e.g. a local server)
    max_samples: int = 15  # member texts shown to the LLM when naming a cluster
    max_label_words: int = 6  # keep labels short and browsable


class TreeConfig(BaseModel):
    """Top-level knobs. Defaults are tuned for the M0/M1 (<100k items) regime.

    Constructed in code only — no environment-variable loading.
    """

    model_config = ConfigDict(extra="ignore", protected_namespaces=())  # allow `model_args` name

    # Defaults are tuned for a human-browsable taxonomy: a small
    # fan-out and small leaves keep every level readable (<=5 sub-topics, <=10
    # items per leaf). Raise both for a large-scale retrieval index instead.
    max_branches: int = 5  # max sub-topics per level (k for KMeans)
    leaf_target: int = 10  # target max items in a leaf before split is attempted
    split_algo: str = "kmeans"  # M0: "kmeans" only
    split_mode: Literal["fixed", "adaptive"] = "fixed"

    # Adaptive split mode treats max_branches as a hard upper bound, then picks
    # the best supported k only when a split materially improves cohesion.
    log_split_decisions: bool = False
    min_cluster_size: int = 2
    min_parent_dispersion: float = 0.08
    parent_dispersion_decay: float = 1.0
    min_split_gain: float = 0.05
    min_split_gain_ratio: float | None = None
    separation_weight: float = 0.05
    imbalance_weight: float = 0.05

    # Distance is always cosine: vectors are L2-normalized and compared with
    # Euclidean (rank-equivalent on the unit sphere). Embeddings encode meaning
    # in direction, not magnitude, so there is no separate distance knob.

    # --- PCA dimensionality reduction -------------------------------------
    # Off by default: only worth it at scale (thousands+). At tens of items PCA
    # is meaningless (too few samples) and never reaches pca_warmup anyway.
    pca_dims: int | None = None  # None = no reduction (operate in raw space)
    pca_mode: Literal["freeze", "incremental"] = "freeze"
    pca_warmup: int = 1000  # items buffered before the first PCA fit
    pca_batch_size: int = 256  # incremental: partial_fit cadence

    rebalance: RebalanceConfig = Field(default_factory=RebalanceConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)  # node auto-naming
    model_args: dict = Field(default_factory=dict)  # passed through to KMeans

    @property
    def leaf_capacity(self) -> int:
        """Deprecated alias for leaf_target."""
        return self.leaf_target

    @property
    def min_samples_to_split(self) -> int:
        """Deprecated alias for leaf_target."""
        return self.leaf_target

    @model_validator(mode="before")
    @classmethod
    def _legacy_leaf_target_aliases(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data

        values = dict(data)
        if "leaf_target" in values:
            return values

        # Backward compatibility:
        # - fixed mode used leaf_capacity as the split threshold;
        # - adaptive mode used min_samples_to_split as the split threshold.
        split_mode = values.get("split_mode", "fixed")
        if split_mode == "adaptive" and "min_samples_to_split" in values:
            values["leaf_target"] = values["min_samples_to_split"]
        elif "leaf_capacity" in values:
            values["leaf_target"] = values["leaf_capacity"]
        elif "min_samples_to_split" in values:
            values["leaf_target"] = values["min_samples_to_split"]
        return values

    @field_validator("max_branches")
    @classmethod
    def _min_branches(cls, v: int) -> int:
        if v < 2:
            raise ValueError("max_branches must be >= 2")
        return v

    @field_validator("leaf_target")
    @classmethod
    def _positive_leaf_target(cls, v: int) -> int:
        if v < 1:
            raise ValueError("leaf_target must be >= 1")
        return v

    @field_validator("split_algo")
    @classmethod
    def _supported_split(cls, v: str) -> str:
        if v != "kmeans":
            raise NotImplementedError(
                f"split_algo={v!r} arrives in a later milestone; "
                "M0 supports 'kmeans' only"
            )
        return v

    @field_validator("min_cluster_size")
    @classmethod
    def _positive_int(cls, v: int) -> int:
        if v < 1:
            raise ValueError("adaptive split integer thresholds must be >= 1")
        return v

    @field_validator(
        "min_parent_dispersion",
        "min_split_gain",
        "min_split_gain_ratio",
        "separation_weight",
        "imbalance_weight",
    )
    @classmethod
    def _nonnegative_float(cls, v: float | None) -> float | None:
        if v is not None and v < 0:
            raise ValueError("adaptive split score thresholds/weights must be >= 0")
        return v

    @field_validator("parent_dispersion_decay")
    @classmethod
    def _decay_factor(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError("parent_dispersion_decay must be between 0 and 1")
        return v

    @model_validator(mode="after")
    def _cross_field(self) -> "TreeConfig":
        if self.pca_dims is not None:
            if self.pca_dims < 2:
                raise ValueError("pca_dims must be >= 2")
            # PCA needs at least n_components samples to fit / partial_fit.
            if self.pca_warmup < self.pca_dims:
                raise ValueError("pca_warmup must be >= pca_dims")
            if self.pca_batch_size < self.pca_dims:
                raise ValueError("pca_batch_size must be >= pca_dims")
        return self
