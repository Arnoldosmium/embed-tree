"""Default representation helpers."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .model import (
    ContentNode,
    KeyNode,
    NodeAggregate,
    NodeEmbedding,
    NodeId,
    PartialTree,
    TreeEdge,
)


DefaultTreeRepresentation = PartialTree


def partial_tree_to_dict(tree: PartialTree) -> dict[str, Any]:
    """Convert a PartialTree to plain JSON-compatible containers."""
    return {
        "content_nodes": [asdict(n) for n in tree.content_nodes],
        "key_nodes": [asdict(n) for n in tree.key_nodes],
        "edges": [asdict(e) for e in tree.edges],
        "embeddings": [asdict(e) for e in tree.embeddings.values()],
        "aggregates": [asdict(a) for a in tree.aggregates.values()],
        "labels": [{"node_id": node_id, "label": label} for node_id, label in tree.labels.items()],
        "embedder_config": tree.embedder_config,
        "projector_state": tree.projector_state,
        "labeler_config": tree.labeler_config,
        "reducer_state": tree.reducer_state,
        "metadata": tree.metadata,
    }


def partial_tree_from_dict(data: dict[str, Any]) -> PartialTree:
    """Build a PartialTree from plain containers."""
    embeddings = [NodeEmbedding(**d) for d in data.get("embeddings", [])]
    aggregates = [NodeAggregate(**d) for d in data.get("aggregates", [])]
    return PartialTree(
        content_nodes=[ContentNode(**d) for d in data.get("content_nodes", [])],
        key_nodes=[KeyNode(**d) for d in data.get("key_nodes", [])],
        edges=[TreeEdge(**d) for d in data.get("edges", [])],
        embeddings={e.node_id: e for e in embeddings},
        aggregates={a.node_id: a for a in aggregates},
        labels={_node_id(d["node_id"]): d["label"] for d in data.get("labels", [])},
        embedder_config=data.get("embedder_config"),
        projector_state=data.get("projector_state"),
        labeler_config=data.get("labeler_config"),
        reducer_state=data.get("reducer_state"),
        metadata=data.get("metadata", {}),
    )


def _node_id(value: Any) -> NodeId:
    return value
