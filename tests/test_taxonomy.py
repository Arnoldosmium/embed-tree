"""Taxonomy: divisive rebuild invariants, labeling, browse, persistence."""

import os
from collections import Counter

import numpy as np

from embed_tree import EmbedTree, FakeEmbeddingProvider, FileTreeStore, TreeConfig

TOPICS = ["animal", "language", "food"]
BASES = {"animal": [1.0, 0.0, 0.0], "language": [0.0, 1.0, 0.0], "food": [0.0, 0.0, 1.0]}


def topical_embedder():
    def embed(text):
        for topic, base in BASES.items():
            if topic in text:
                h = (abs(hash(text)) % 1000) / 1000.0
                return np.array(base) + 0.01 * h
        return np.zeros(3)

    return embed


def topic_of(text):
    return next(t for t in TOPICS if t in text)


def topic_tagger(texts):
    words = [topic_of(t) for t in texts if any(t2 in t for t2 in TOPICS)]
    return Counter(words).most_common(1)[0][0] if words else ""


def build_topical_tree(per=10):
    tree = EmbedTree(
        embedder=topical_embedder(),
        config=TreeConfig(max_branches=3, leaf_capacity=10, model_args={"random_state": 0}),
    )
    for topic in TOPICS:
        for i in range(per):
            tree.add(f"{topic} note {i}")
    return tree


def _invariants(node, cfg):
    if node.is_leaf:
        assert len(node.items or []) <= cfg.leaf_capacity or node.unsplittable
    else:
        assert len(node.children) <= cfg.max_branches
        assert len(node.children) >= 2
        for c in node.children:
            _invariants(c, cfg)


def test_divisive_invariants_and_count():
    cfg = TreeConfig(max_branches=5, leaf_capacity=10, model_args={"random_state": 0})
    tree = EmbedTree(embedder=FakeEmbeddingProvider(dim=12), config=cfg)
    tree.add_batch([f"doc-{i}" for i in range(47)])
    tree.rebalance()
    assert len(tree) == 47
    _invariants(tree.get_tree(), cfg)


def test_leaves_are_topic_homogeneous():
    tree = build_topical_tree()
    tree.rebalance()
    root = tree.get_tree()
    assert not root.is_leaf and len(root.children) == 3
    for leaf in root.children:
        assert leaf.is_leaf and leaf.count == 10
        topics = {topic_of(it.text) for it in leaf.items}
        assert len(topics) == 1, topics  # clustering grouped one topic per leaf


def test_labels_via_injected_tagger():
    tree = build_topical_tree()
    tree.organize(tagger=topic_tagger)
    root = tree.get_tree()
    assert root.label in TOPICS  # root labeled with dominant topic
    leaf_labels = {leaf.label for leaf in root.children}
    assert leaf_labels == set(TOPICS)  # each leaf named by its topic


def test_browse_dict_and_show():
    tree = build_topical_tree()
    tree.organize(tagger=topic_tagger)

    d = tree.to_dict(max_items=2)
    assert d["size"] == 30 and "children" in d
    leaf = d["children"][0]
    assert "items" in leaf and len(leaf["items"]) <= 2
    assert leaf["label"] in TOPICS

    text = tree.show(max_items=2)
    assert any(topic in text for topic in TOPICS)
    assert "[30]" in text  # root size shown


def test_single_leaf_branch_is_lazy_labeled_and_collapsible():
    tree = build_topical_tree()
    tree.rebalance()
    tree.remove_batch([*range(10), *range(20, 30)])  # keep only the language leaf

    calls = []

    def counting_tagger(texts):
        calls.append(texts)
        return "should not be used"

    tree.label(tagger=counting_tagger)
    assert calls == []  # no LLM/tagger call for a non-branching wrapper + sole leaf

    d = tree.to_dict(max_items=2, collapse_single_leaf=True)
    assert "items" in d and "children" not in d
    assert d["size"] == 10
    assert d["label"].startswith("language note")

    text = tree.show(max_items=1, collapse_single_leaf=True)
    assert "(unlabeled)" not in text
    assert "language note" in text


def test_organize_persistence_keeps_labels(tmp_path):
    path = os.path.join(tmp_path, "tax.json")
    t1 = EmbedTree(
        embedder=topical_embedder(),
        store=FileTreeStore(path),
        config=TreeConfig(max_branches=3, leaf_capacity=10, model_args={"random_state": 0}),
    )
    for topic in TOPICS:
        for i in range(10):
            t1.add(f"{topic} note {i}")
    t1.organize(tagger=topic_tagger)
    labels_before = sorted(c.label for c in t1.get_tree().children)

    t2 = EmbedTree(
        embedder=topical_embedder(),
        store=FileTreeStore(path),
        config=TreeConfig(max_branches=3, leaf_capacity=10),
    )
    assert len(t2) == 30
    labels_after = sorted(c.label for c in t2.get_tree().children)
    assert labels_before == labels_after  # labels survived reload


def test_keyword_tagger_default_when_no_llm():
    # No tagger injected, provider "none" -> KeywordTagger names nodes.
    tree = build_topical_tree()
    tree.organize()  # uses config.llm (provider none) -> TF-IDF
    for leaf in tree.get_tree().children:
        assert leaf.label  # non-empty keyword label
