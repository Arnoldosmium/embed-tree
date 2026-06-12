"""M0 behavior tests: add, split, query, persistence round-trip."""

import os

import numpy as np
import pytest

from embed_tree import EmbedTree, JsonTreeLoader, TreeConfig


def make_clustered_embedder(seed=0, dim=8, spread=0.05):
    """Content is (cluster_id, idx); vectors tightly cluster around 5 centers."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(5, dim))

    def embed(content):
        cluster_id, idx = content
        local = np.random.default_rng((cluster_id + 1) * 1000 + idx)
        return centers[cluster_id] + local.normal(scale=spread, size=dim)

    return embed


def test_add_and_len():
    tree = EmbedTree(embedder=make_clustered_embedder(), config=TreeConfig(leaf_capacity=20, max_branches=5))
    for c in range(5):
        for i in range(10):
            tree.add((c, i))
    assert len(tree) == 50


def test_split_creates_internal_nodes():
    # 200 items with leaf_capacity 20 must force at least one split.
    tree = EmbedTree(embedder=make_clustered_embedder(), config=TreeConfig(leaf_capacity=20, max_branches=5))
    for c in range(5):
        for i in range(40):
            tree.add((c, i))
    assert not tree.get_tree().is_leaf, "root should have split"
    assert len(tree) == 200


def test_query_finds_same_cluster():
    embed = make_clustered_embedder()
    cfg = TreeConfig(leaf_capacity=20, max_branches=5, model_args={"random_state": 0})
    tree = EmbedTree(embedder=embed, config=cfg)
    for c in range(5):
        for i in range(40):
            tree.add((c, i), payload={"cluster": c})
    tree.rebalance()  # clean, balanced divisive tree -> homogeneous leaves

    # Query near cluster 2; top hits should be from cluster 2.
    hits = tree.query((2, 999), k=5)
    assert hits, "expected results"
    top_clusters = [p["cluster"] for _, _, p in hits]
    assert top_clusters.count(2) >= 4, top_clusters


def test_exhaustive_matches_brute_force():
    embed = make_clustered_embedder()
    cfg = TreeConfig(leaf_capacity=15, max_branches=4)
    tree = EmbedTree(embedder=embed, config=cfg)
    for c in range(5):
        for i in range(30):
            tree.add((c, i))
    hits = tree.query((2, 999), k=1, exhaustive=True)
    assert len(hits) == 1


def test_persistence_round_trip(tmp_path):
    path = os.path.join(tmp_path, "tree.json")
    embed = make_clustered_embedder()
    cfg = TreeConfig(leaf_capacity=20, max_branches=5)

    t1 = EmbedTree(embedder=embed, state=JsonTreeLoader(path), config=cfg)
    for c in range(5):
        for i in range(40):
            t1.add((c, i), payload={"cluster": c})
    before = t1.query((3, 999), k=5)

    # Reload from disk into a fresh instance.
    t2 = EmbedTree(embedder=embed, state=JsonTreeLoader(path), config=cfg)
    assert len(t2) == 200
    after = t2.query((3, 999), k=5)
    assert [h[0] for h in before] == [h[0] for h in after]


def test_duplicate_vectors_are_unsplittable():
    # All identical vectors: KMeans can't split; tree must not loop forever.
    def embed(_):
        return np.ones(4)

    tree = EmbedTree(embedder=embed, config=TreeConfig(leaf_capacity=10, max_branches=3))
    for i in range(50):
        tree.add(i)
    assert len(tree) == 50
    assert tree.get_tree().is_leaf
    assert tree.get_tree().unsplittable


def test_rebalance_preserves_items():
    embed = make_clustered_embedder()
    cfg = TreeConfig(leaf_capacity=20, max_branches=5)
    tree = EmbedTree(embedder=embed, config=cfg)
    for c in range(5):
        for i in range(30):
            tree.add((c, i))
    n = len(tree)
    tree.rebalance()
    assert len(tree) == n


def _check_counts(node):
    """Every node.count and vsum must match the items actually beneath it."""
    if node.is_leaf:
        items = list(node.items or [])
    else:
        items = [it for c in node.children for it in _check_counts(c)]
    assert node.count == len(items)
    if items:
        assert np.allclose(node.vsum, np.sum([it.vector for it in items], axis=0), atol=1e-9)
    return items


def test_remove_keeps_centroids_consistent():
    embed = make_clustered_embedder()
    cfg = TreeConfig(leaf_capacity=20, max_branches=5)
    tree = EmbedTree(embedder=embed, config=cfg)
    ids = [tree.add((c, i)) for c in range(5) for i in range(40)]

    assert tree.remove(ids[0]) is True
    assert tree.remove(ids[0]) is False  # already gone
    assert tree.remove(10**9) is False  # never existed

    removed = tree.remove_batch(ids[1:100])
    assert removed == 99
    assert len(tree) == 200 - 100
    _check_counts(tree.get_tree())  # vsum/count rolled back along every ancestor


def test_remove_prunes_empty_leaves():
    embed = make_clustered_embedder()
    cfg = TreeConfig(leaf_capacity=20, max_branches=5)
    tree = EmbedTree(embedder=embed, config=cfg)
    ids = [tree.add((c, i)) for c in range(5) for i in range(40)]
    tree.rebalance()
    n_leaves = lambda node: 1 if node.is_leaf else sum(n_leaves(c) for c in node.children)
    before = n_leaves(tree.get_tree())
    for iid in ids:  # empty the tree entirely
        tree.remove(iid)
    assert len(tree) == 0
    assert n_leaves(tree.get_tree()) < before  # dead leaves were pruned
    assert tree.get_tree().is_leaf  # collapsed back to an empty root


def test_pca_config_validation():
    with pytest.raises(Exception):
        TreeConfig(pca_dims=50, pca_warmup=10)  # warmup < pca_dims
    with pytest.raises(Exception):
        TreeConfig(pca_dims=1)  # too small
    TreeConfig(pca_dims=8, pca_warmup=50, pca_batch_size=16)  # ok
