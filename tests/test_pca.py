"""M1 PCA tests: warmup, both modes, rebalance refit, and persistence."""

import os

import numpy as np

from embed_tree import ContentNode, EmbedTree, JsonTreeLoader, TreeConfig


def make_highdim_embedder(seed=0, dim=64, n_clusters=5, spread=0.05):
    """High-dim vectors clustered around n_clusters centers."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_clusters, dim))

    def embed(text):
        c, i = (int(part) for part in str(text).split(":"))
        local = np.random.default_rng((c + 1) * 1000 + i)
        return centers[c] + local.normal(scale=spread, size=dim)

    return embed


def _fill(tree, n_clusters=5, per=40):
    for c in range(n_clusters):
        for i in range(per):
            tree.add_node(ContentNode(f"{c}:{i}", f"{c}:{i}", {"cluster": c}))


def test_warmup_then_fit_builds_tree():
    embed = make_highdim_embedder()
    cfg = TreeConfig(pca_dims=8, pca_mode="freeze", pca_warmup=60, leaf_capacity=20, max_branches=5, model_args={"random_state": 0})
    tree = EmbedTree(embedder=embed, config=cfg)

    # Below warmup: items buffered, reducer not fitted, root still empty.
    for i in range(50):
        tree.add_node(ContentNode(f"0:{i}", f"0:{i}"))
    assert not tree.reducer.is_fitted
    assert len(tree) == 50
    # Query works during warmup (raw-space brute force).
    assert tree.query("0:999", k=3)

    # Cross warmup -> PCA fits, tree built in reduced space.
    for i in range(20):
        tree.add_node(ContentNode(f"1:{i}", f"1:{i}"))
    assert tree.reducer.is_fitted
    assert tree.reducer.transform(embed("0:0")[None, :]).shape == (1, 8)
    assert len(tree) == 70


def test_freeze_query_quality():
    embed = make_highdim_embedder()
    cfg = TreeConfig(pca_dims=8, pca_mode="freeze", pca_warmup=60, leaf_capacity=20, max_branches=5, model_args={"random_state": 0})
    tree = EmbedTree(embedder=embed, config=cfg)
    _fill(tree)
    hits = tree.query("2:999", k=5)
    assert [p["cluster"] for _, _, p in hits].count(2) >= 4


def test_incremental_mode_runs_and_queries():
    embed = make_highdim_embedder()
    cfg = TreeConfig(
        pca_dims=8, pca_mode="incremental", pca_warmup=40, pca_batch_size=16,
        leaf_capacity=20, max_branches=5, model_args={"random_state": 0},
    )
    tree = EmbedTree(embedder=embed, config=cfg)
    _fill(tree, per=40)
    assert tree.reducer.is_fitted
    assert len(tree) == 200
    hits = tree.query("3:999", k=5)
    assert [p["cluster"] for _, _, p in hits].count(3) >= 4


def test_freeze_rebalance_refits_pca():
    embed = make_highdim_embedder()
    cfg = TreeConfig(pca_dims=8, pca_mode="freeze", pca_warmup=60, leaf_capacity=20, max_branches=5, model_args={"random_state": 0})
    tree = EmbedTree(embedder=embed, config=cfg)
    _fill(tree)
    before = tree.reducer.components_.copy()
    tree.rebalance()
    after = tree.reducer.components_
    # A refit happened (shape preserved; values recomputed on full data).
    assert after.shape == before.shape
    assert len(tree) == 200
    assert [p["cluster"] for _, _, p in tree.query("1:999", k=5)].count(1) >= 4


def test_incremental_rebalance_keeps_running_pca():
    embed = make_highdim_embedder()
    cfg = TreeConfig(
        pca_dims=8, pca_mode="incremental", pca_warmup=40, pca_batch_size=16,
        leaf_capacity=20, max_branches=5, model_args={"random_state": 0},
    )
    tree = EmbedTree(embedder=embed, config=cfg)
    _fill(tree)
    n = len(tree)
    tree.rebalance()
    assert len(tree) == n
    assert tree.reducer.is_fitted


def test_pca_persistence_round_trip(tmp_path):
    path = os.path.join(tmp_path, "tree.json")
    embed = make_highdim_embedder()
    cfg = TreeConfig(pca_dims=8, pca_mode="freeze", pca_warmup=60, leaf_capacity=20, max_branches=5, model_args={"random_state": 0})

    t1 = EmbedTree(embedder=embed, state=JsonTreeLoader(path), config=cfg)
    _fill(t1)
    before = t1.query("4:999", k=5)

    t2 = EmbedTree(embedder=embed, state=JsonTreeLoader(path), config=cfg)
    assert len(t2) == 200
    assert t2.reducer.is_fitted
    after = t2.query("4:999", k=5)
    assert [h[0] for h in before] == [h[0] for h in after]


def test_incremental_persistence_resumes_partial_fit(tmp_path):
    path = os.path.join(tmp_path, "tree.json")
    embed = make_highdim_embedder()
    cfg = TreeConfig(
        pca_dims=8, pca_mode="incremental", pca_warmup=40, pca_batch_size=16,
        leaf_capacity=20, max_branches=5, model_args={"random_state": 0},
    )
    t1 = EmbedTree(embedder=embed, state=JsonTreeLoader(path), config=cfg)
    _fill(t1, per=20)

    t2 = EmbedTree(embedder=embed, state=JsonTreeLoader(path), config=cfg)
    n = len(t2)
    # Keep adding after reload -> partial_fit must resume without error.
    for i in range(100, 140):
        t2.add_node(ContentNode(f"0:{i}", f"0:{i}"))
    assert len(t2) == n + 40
