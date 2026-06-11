"""Default tree reconciler."""

from __future__ import annotations

from typing import Any, Callable

from embed_tree.persisters.model import MaterializedTreeState
from embed_tree.loaders import TreeLoader
from embed_tree.representation import ContentNode, PartialTree


class DefaultTreeReconciler:
    """Reconcile ground truth with optional reusable loaded state.

    This default reconciler is intentionally representation-level only. It
    removes reusable records whose ids are no longer in ground truth and reuses
    embeddings/labels/aggregates for ids that still exist. Operational tree
    construction remains the responsibility of ``EmbedTree`` or a future
    materializer.
    """

    def reconcile(
        self,
        ground_truth_loader: TreeLoader,
        reusable_loader: TreeLoader | None = None,
        *,
        embedder: Callable[[Any], Any],
        config: Any | None = None,
    ) -> PartialTree | MaterializedTreeState:
        ground_truth = ground_truth_loader.load()
        if ground_truth is None:
            ground_truth = PartialTree()
        reusable = reusable_loader.load() if reusable_loader is not None else None
        cached = reusable if isinstance(reusable, PartialTree) else PartialTree()
        ids = {node.id for node in ground_truth.content_nodes} | {node.id for node in ground_truth.key_nodes}
        tree = PartialTree(
            content_nodes=list(ground_truth.content_nodes),
            key_nodes=list(ground_truth.key_nodes),
            edges=[edge for edge in (ground_truth.edges or cached.edges) if edge.parent_id in ids and edge.child_id in ids],
            embeddings={node_id: emb for node_id, emb in cached.embeddings.items() if node_id in ids},
            aggregates={node_id: agg for node_id, agg in cached.aggregates.items() if node_id in ids},
            labels={node_id: label for node_id, label in cached.labels.items() if node_id in ids},
            embedder_config=ground_truth.embedder_config or cached.embedder_config,
            projector_state=cached.projector_state,
            labeler_config=ground_truth.labeler_config or cached.labeler_config,
            reducer_state=cached.reducer_state,
            metadata={**cached.metadata, **ground_truth.metadata},
        )

        for node in tree.content_nodes:
            if node.id not in tree.embeddings:
                vector = embedder(_content_for_embedding(node))
                tree.embeddings[node.id] = _embedding(node.id, vector)

        return tree


def _content_for_embedding(node: ContentNode) -> Any:
    return node.content


def _embedding(node_id: Any, vector: Any) -> Any:
    from embed_tree.representation import NodeEmbedding

    return NodeEmbedding(node_id=node_id, vector=list(vector))
