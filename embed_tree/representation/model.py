"""Storage-neutral tree representation models.

These models are deliberately smaller and looser than the live ``Node`` /
``Item`` objects used by ``EmbedTree``. Loaders return this partial
representation; a reconciler/materializer decides what can be reused and what
must be recomputed before the tree becomes operational.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Hashable, Sequence

NodeId = Hashable
VectorData = Sequence[float]


@dataclass(frozen=True)
class ContentNode:
    """Authoritative leaf content supplied by a loader."""

    id: NodeId
    content: Any
    text: str | None = None
    payload: Any = None
    version: str | None = None


@dataclass(frozen=True)
class KeyNode:
    """Optional non-leaf node supplied by a loader."""

    id: NodeId
    label: str | None = None
    payload: Any = None
    version: str | None = None


@dataclass(frozen=True)
class TreeEdge:
    """Parent -> child relationship in a partial tree."""

    parent_id: NodeId
    child_id: NodeId


@dataclass(frozen=True)
class NodeEmbedding:
    """Reusable embedding data for a content or key node."""

    node_id: NodeId
    vector: VectorData
    raw: VectorData | None = None
    model: str | None = None
    version: str | None = None


@dataclass(frozen=True)
class NodeAggregate:
    """Reusable derived cache for a subtree."""

    node_id: NodeId
    vsum: VectorData
    count: int
    version: str | None = None


@dataclass
class PartialTree:
    """A loader-produced tree fragment.

    ``content_nodes`` are the only required source records. Everything else is
    optional reusable state: shape, embeddings, labels, aggregates, reducer
    state, or implementation-specific metadata.
    """

    content_nodes: list[ContentNode] = field(default_factory=list)
    key_nodes: list[KeyNode] = field(default_factory=list)
    edges: list[TreeEdge] = field(default_factory=list)
    embeddings: dict[NodeId, NodeEmbedding] = field(default_factory=dict)
    aggregates: dict[NodeId, NodeAggregate] = field(default_factory=dict)
    labels: dict[NodeId, str] = field(default_factory=dict)
    embedder_config: dict[str, Any] | None = None
    projector_state: dict[str, Any] | None = None
    labeler_config: dict[str, Any] | None = None
    reducer_state: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
