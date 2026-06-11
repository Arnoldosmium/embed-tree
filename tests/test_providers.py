"""Provider tests: fake determinism, caching, normalize, batching, and the
OpenAI adapter via an injected stub client (no network)."""

import numpy as np
import pytest

from embed_tree import EmbedTree, FakeEmbeddingProvider, OpenAIEmbeddingProvider, TreeConfig


def test_fake_is_deterministic():
    p = FakeEmbeddingProvider(dim=16)
    assert np.array_equal(p("hello"), p("hello"))
    assert not np.array_equal(p("hello"), p("world"))


def test_callable_shape():
    p = FakeEmbeddingProvider(dim=16)
    assert p("x").shape == (16,)
    assert p.embed_batch(["a", "b", "c"]).shape == (3, 16)


def test_normalize_option():
    p = FakeEmbeddingProvider(dim=16, normalize=True)
    assert np.allclose(np.linalg.norm(p("hello")), 1.0)


def test_cache_avoids_recompute():
    calls = {"n": 0}

    class Counting(FakeEmbeddingProvider):
        def _embed_batch(self, texts):
            calls["n"] += len(texts)
            return super()._embed_batch(texts)

    p = Counting(dim=8, cache=True)
    p("a"); p("a"); p("a")
    assert calls["n"] == 1  # embedded once, then served from cache
    p.cache_clear()
    p("a")
    assert calls["n"] == 2


def test_cache_partial_batch_order_preserved():
    p = FakeEmbeddingProvider(dim=8)
    p("b")  # warm one entry
    out = p.embed_batch(["a", "b", "c"])
    assert np.array_equal(out[1], p("b"))
    assert np.array_equal(out[0], p("a"))


def test_retry_then_succeed():
    attempts = {"n": 0}

    class Flaky(FakeEmbeddingProvider):
        def _embed_batch(self, texts):
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise RuntimeError("429")
            return super()._embed_batch(texts)

    p = Flaky(dim=4, max_retries=5, backoff_base=0.0)
    assert p("x").shape == (4,)
    assert attempts["n"] == 3


def test_provider_plugs_into_tree():
    tree = EmbedTree(
        embedder=FakeEmbeddingProvider(dim=24),
        config=TreeConfig(leaf_capacity=20, max_branches=4),
    )
    for i in range(60):
        tree.add(f"doc-{i}")
    assert len(tree) == 60


# --- OpenAI adapter via injected stub client (no network) ------------------

class _StubEmbeddings:
    def create(self, model, input, **kwargs):
        class _D:
            def __init__(self, e):
                self.embedding = e

        class _R:
            pass

        r = _R()
        # echo deterministic vectors keyed by text length
        r.data = [_D([float(len(t)), 1.0, 2.0]) for t in input]
        return r


class _StubClient:
    def __init__(self):
        self.embeddings = _StubEmbeddings()


def test_openai_adapter_uses_client_and_batches():
    p = OpenAIEmbeddingProvider(client=_StubClient(), model="text-embedding-3-small")
    out = p.embed_batch(["a", "bb", "ccc"])
    assert out.shape == (3, 3)
    assert out[0, 0] == 1.0 and out[2, 0] == 3.0  # len-based echo, order preserved
