"""Serialization helpers for public tree representations."""

from __future__ import annotations

from typing import Any

from .model import BranchNode, ContentNode, NodeId

DefaultTreeRepresentation = BranchNode


def tree_to_dict(node: BranchNode | ContentNode) -> dict[str, Any]:
    if isinstance(node, ContentNode):
        return {
            "kind": "content",
            "id": node.id,
            "text": node.text,
            "metadata": node.metadata,
            "embedding": None if node.embedding is None else list(node.embedding),
        }
    return {
        "kind": "branch",
        "id": node.id,
        "label": node.label,
        "children": [tree_to_dict(child) for child in node.children],
        "metadata": node.metadata,
        "vector_sum": None if node.vector_sum is None else list(node.vector_sum),
        "count": node.count,
    }


def tree_from_dict(data: dict[str, Any]) -> BranchNode | ContentNode:
    if data.get("kind") == "content":
        return ContentNode(
            id=_node_id(data["id"]),
            text=data["text"],
            metadata=data.get("metadata", {}),
            embedding=data.get("embedding"),
        )
    return BranchNode(
        id=_node_id(data["id"]),
        label=data.get("label"),
        children=[tree_from_dict(child) for child in data.get("children", [])],
        metadata=data.get("metadata", {}),
        vector_sum=data.get("vector_sum"),
        _count=data.get("count"),
    )


def _node_id(value: Any) -> NodeId:
    return value
