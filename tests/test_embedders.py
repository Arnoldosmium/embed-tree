"""Text embedder tests."""

import numpy as np

from embed_tree import ContentNode, EmbedTree, OpenAITextEmbedder, TagSetEmbedder, TreeConfig
from tests.helpers import FakeTextEmbedder


def test_fake_is_deterministic():
    embedder = FakeTextEmbedder(dim=16)
    assert np.array_equal(embedder("hello"), embedder("hello"))
    assert not np.array_equal(embedder("hello"), embedder("world"))


def test_callable_shape():
    embedder = FakeTextEmbedder(dim=16)
    assert embedder("x").shape == (16,)
    assert embedder.embed_batch(["a", "b", "c"]).shape == (3, 16)


def test_normalize_option():
    embedder = FakeTextEmbedder(dim=16, normalize=True)
    assert np.allclose(np.linalg.norm(embedder("hello")), 1.0)


def test_cache_avoids_recompute():
    calls = {"n": 0}

    class Counting(FakeTextEmbedder):
        def _embed_batch(self, texts):
            calls["n"] += len(texts)
            return super()._embed_batch(texts)

    embedder = Counting(dim=8, cache=True)
    embedder("a")
    embedder("a")
    embedder("a")
    assert calls["n"] == 1
    embedder.cache_clear()
    embedder("a")
    assert calls["n"] == 2


def test_cache_partial_batch_order_preserved():
    embedder = FakeTextEmbedder(dim=8)
    embedder("b")
    out = embedder.embed_batch(["a", "b", "c"])
    assert np.array_equal(out[1], embedder("b"))
    assert np.array_equal(out[0], embedder("a"))


def test_retry_then_succeed():
    attempts = {"n": 0}

    class Flaky(FakeTextEmbedder):
        def _embed_batch(self, texts):
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise RuntimeError("429")
            return super()._embed_batch(texts)

    embedder = Flaky(dim=4, max_retries=5, backoff_base=0.0)
    assert embedder("x").shape == (4,)
    assert attempts["n"] == 3


def test_embedder_plugs_into_tree():
    tree = EmbedTree(
        embedder=FakeTextEmbedder(dim=24),
        config=TreeConfig(leaf_capacity=20, max_branches=4),
    )
    for i in range(60):
        tree.add_node(ContentNode(i, f"doc-{i}"))
    assert len(tree) == 60


class _StubEmbeddings:
    def create(self, model, input, **kwargs):
        class _D:
            def __init__(self, embedding):
                self.embedding = embedding

        class _R:
            pass

        response = _R()
        response.data = [_D([float(len(text)), 1.0, 2.0]) for text in input]
        return response


class _StubClient:
    def __init__(self):
        self.embeddings = _StubEmbeddings()


def test_openai_text_embedder_uses_client_and_batches():
    embedder = OpenAITextEmbedder(client=_StubClient(), model="text-embedding-3-small")
    out = embedder.embed_batch(["a", "bb", "ccc"])
    assert out.shape == (3, 3)
    assert out[0, 0] == 1.0 and out[2, 0] == 3.0


def test_tag_set_embedder_multihot():
    embedder = TagSetEmbedder(["docs", "ingest", "analysis"])

    assert np.array_equal(embedder(["docs", "analysis"]), np.asarray([1.0, 0.0, 1.0], dtype=np.float32))
    assert np.array_equal(embedder({"tags": ["ingest"]}), np.asarray([0.0, 1.0, 0.0], dtype=np.float32))
    assert np.array_equal(embedder("docs"), np.asarray([1.0, 0.0, 0.0], dtype=np.float32))


def test_tag_set_embedder_unknown_error():
    embedder = TagSetEmbedder(["docs"], unknown="error")

    try:
        embedder(["missing"])
    except ValueError as exc:
        assert "unknown tag" in str(exc)
    else:
        raise AssertionError("expected ValueError")
