"""add_batch: batched embedding (one backend call), single persist, alignment."""

import numpy as np

from embed_tree import EmbedTree, TreeConfig
from tests.helpers import FakeTextEmbedder


def test_add_batch_returns_ids_and_inserts():
    tree = EmbedTree(embedder=FakeTextEmbedder(dim=16), config=TreeConfig(leaf_capacity=20, max_branches=4))
    ids = tree.add_batch([f"doc-{i}" for i in range(50)])
    assert len(ids) == 50
    assert len(set(ids)) == 50
    assert len(tree) == 50


def test_add_batch_uses_single_backend_call():
    calls = {"n": 0}

    class Counting(FakeTextEmbedder):
        def _embed_batch(self, texts):
            calls["n"] += 1
            return super()._embed_batch(texts)

    tree = EmbedTree(embedder=Counting(dim=16), config=TreeConfig(leaf_capacity=100, max_branches=4))
    tree.add_batch([f"doc-{i}" for i in range(30)])
    assert calls["n"] == 1  # one batched call, not 30


def test_add_batch_persists_once():
    saves = {"n": 0}

    class CountingState:
        def load(self):
            return None

        def save(self, state):
            saves["n"] += 1

    tree = EmbedTree(
        embedder=FakeTextEmbedder(dim=16),
        state=CountingState(),
        config=TreeConfig(leaf_capacity=100, max_branches=4),
    )
    tree.add_batch([f"doc-{i}" for i in range(40)])
    assert saves["n"] == 1  # single snapshot, not 40


def test_add_batch_payload_and_id_alignment():
    tree = EmbedTree(embedder=FakeTextEmbedder(dim=16), config=TreeConfig(leaf_capacity=100, max_branches=4))
    ids = tree.add_batch(
        ["a", "b", "c"],
        item_ids=[10, 20, 30],
        payloads=[{"k": "a"}, {"k": "b"}, {"k": "c"}],
    )
    assert ids == [10, 20, 30]
    hit = tree.query("a", k=1)[0]
    assert hit[0] == 10 and hit[2] == {"k": "a"}


def test_add_batch_accepts_string_item_ids():
    tree = EmbedTree(embedder=FakeTextEmbedder(dim=16), config=TreeConfig(leaf_capacity=100, max_branches=4))
    ids = tree.add_batch(["a", "b"], item_ids=["md5-a", "md5-b"])

    assert ids == ["md5-a", "md5-b"]
    assert tree.query("a", k=1, exhaustive=True)[0][0] == "md5-a"
    assert tree.remove("md5-a") is True
    assert tree.remove("md5-a") is False


def test_add_batch_falls_back_for_plain_callable():
    # A plain callable without embed_batch must still work.
    embed = lambda c: np.array([float(len(c)), 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    tree = EmbedTree(embedder=embed, config=TreeConfig(leaf_capacity=100, max_branches=4))
    ids = tree.add_batch(["a", "bb", "ccc"])
    assert len(ids) == 3 and len(tree) == 3


def test_add_batch_crosses_pca_warmup():
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(4, 32))

    def embed(content):
        c, i = content
        return centers[c] + np.random.default_rng(c * 1000 + i).normal(scale=0.05, size=32)

    cfg = TreeConfig(pca_dims=8, pca_mode="freeze", pca_warmup=40, leaf_capacity=20, max_branches=4)
    tree = EmbedTree(embedder=embed, config=cfg)
    tree.add_batch([(c, i) for c in range(4) for i in range(20)])  # 80 > warmup
    assert tree.reducer.is_fitted
    assert len(tree) == 80
