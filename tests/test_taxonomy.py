"""Taxonomy: divisive rebuild invariants, labeling, browse, persistence."""

import logging
import os
from collections import Counter

import numpy as np

from embed_tree import BranchNode, ContentNode, EmbedTree, FunctionLabeler, JsonTreeLoader, LabelRequest, TreeConfig
from tests.helpers import FakeTextEmbedder

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


def topic_labeler(request: LabelRequest) -> str:
    texts = [candidate.text for candidate in request.candidates]
    words = [topic_of(text) for text in texts if any(topic in text for topic in TOPICS)]
    return Counter(words).most_common(1)[0][0] if words else ""


def build_topical_tree(per=10):
    tree = EmbedTree(
        embedder=topical_embedder(),
        config=TreeConfig(max_branches=3, leaf_capacity=10, model_args={"random_state": 0}),
    )
    for topic in TOPICS:
        for i in range(per):
            tree.add_node(ContentNode(f"{topic}:{i}", f"{topic} note {i}"))
    return tree


def _invariants(node, cfg):
    if node.is_leaf:
        assert len(node.items or []) <= cfg.leaf_target or node.unsplittable
    else:
        assert len(node.children) <= cfg.max_branches
        assert len(node.children) >= 2
        for c in node.children:
            _invariants(c, cfg)


def test_divisive_invariants_and_count():
    cfg = TreeConfig(max_branches=5, leaf_capacity=10, model_args={"random_state": 0})
    tree = EmbedTree(embedder=FakeTextEmbedder(dim=12), config=cfg)
    tree.add_nodes([ContentNode(i, f"doc-{i}") for i in range(47)])
    tree.rebalance()
    assert len(tree) == 47
    _invariants(tree.root, cfg)


def test_leaves_are_topic_homogeneous():
    tree = build_topical_tree()
    tree.rebalance()
    root = tree.root
    assert not root.is_leaf and len(root.children) == 3
    for leaf in root.children:
        assert leaf.is_leaf and leaf.count == 10
        topics = {topic_of(it.text) for it in leaf.items}
        assert len(topics) == 1, topics  # clustering grouped one topic per leaf


def test_adaptive_rebalance_discovers_natural_branches():
    def embed(text):
        return np.array(BASES[topic_of(text)])

    tree = EmbedTree(
        embedder=embed,
        config=TreeConfig(
            split_mode="adaptive",
            max_branches=5,
            leaf_target=6,
            min_cluster_size=3,
            min_split_gain=0.01,
            model_args={"random_state": 0},
        ),
    )
    for topic in TOPICS:
        for i in range(10):
            tree.add_node(ContentNode(f"{topic}:{i}", f"{topic} note {i}"))

    tree.rebalance()

    assert not tree.root.is_leaf
    assert len(tree.root.children) == 3
    for child in tree.root.children:
        assert child.is_leaf
        assert {topic_of(it.text) for it in child.items} in ({topic} for topic in TOPICS)


def test_adaptive_rebalance_keeps_coherent_large_branch_together():
    def embed(_):
        return np.ones(4)

    tree = EmbedTree(
        embedder=embed,
        config=TreeConfig(
            split_mode="adaptive",
            max_branches=5,
            leaf_target=6,
            min_parent_dispersion=0.01,
            model_args={"random_state": 0},
        ),
    )
    for i in range(30):
        tree.add_node(ContentNode(i, f"same topic {i}"))

    tree.rebalance()

    assert tree.root.is_leaf
    assert tree.root.count == 30


def test_adaptive_split_verbose_logs_decision_details(caplog):
    def embed(text):
        return np.array(BASES[topic_of(text)])

    tree = EmbedTree(
        embedder=embed,
        config=TreeConfig(
            split_mode="adaptive",
            log_split_decisions=True,
            max_branches=3,
            leaf_target=6,
            min_cluster_size=3,
            min_split_gain=0.01,
            model_args={"random_state": 0},
        ),
    )
    for topic in TOPICS:
        for i in range(4):
            tree.add_node(ContentNode(f"{topic}:{i}", f"{topic} note {i}"))

    caplog.set_level(logging.INFO, logger="embed_tree.splitters")
    tree.rebalance()

    assert "split.adaptive.candidate" in caplog.text
    assert "split.adaptive.accept" in caplog.text
    assert "parent_dispersion=" in caplog.text
    assert "gain=" in caplog.text
    assert "gain_ratio=" in caplog.text
    assert "depth=" in caplog.text
    assert "sizes=" in caplog.text


def test_adaptive_parent_dispersion_decay_allows_deeper_splits():
    vectors = {}
    for top in ("a", "b"):
        for sub in ("x", "y"):
            for i in range(8):
                if top == "a":
                    vector = np.array([1.0, 0.12 if sub == "x" else -0.12, 0.0, 0.0])
                else:
                    vector = np.array([0.0, 1.0, 0.0, 0.12 if sub == "x" else -0.12])
                vectors[f"{top}:{sub}:{i}"] = vector

    def embed(text):
        return vectors[text]

    base = dict(
        split_mode="adaptive",
        leaf_target=4,
        max_branches=2,
        min_cluster_size=2,
        min_parent_dispersion=0.15,
        min_split_gain=0.0,
        model_args={"random_state": 0},
    )
    flat = EmbedTree(embedder=embed, config=TreeConfig(**base))
    decayed = EmbedTree(embedder=embed, config=TreeConfig(**base, parent_dispersion_decay=0.7))
    for key in vectors:
        node = ContentNode(key, key)
        flat.add_node(node)
        decayed.add_node(node)

    flat.rebalance()
    decayed.rebalance()

    assert [child.count for child in flat.root.children or []] == [16, 16]
    assert all(child.is_leaf for child in flat.root.children or [])
    assert [len(child.children or []) for child in decayed.root.children or []] == [2, 2]


def test_adaptive_relative_gain_can_accept_low_absolute_gain():
    vectors = {}
    for sub, sign in (("x", 1), ("y", -1)):
        for i in range(6):
            vectors[f"{sub}:{i}"] = np.array([1.0, sign * 0.02, 0.0, 0.0])

    def embed(text):
        return vectors[text]

    base = dict(
        split_mode="adaptive",
        leaf_target=4,
        max_branches=2,
        min_cluster_size=2,
        min_parent_dispersion=0.001,
        min_split_gain=0.05,
        model_args={"random_state": 0},
    )
    absolute_only = EmbedTree(embedder=embed, config=TreeConfig(**base))
    relative = EmbedTree(embedder=embed, config=TreeConfig(**base, min_split_gain_ratio=0.5))
    for key in vectors:
        node = ContentNode(key, key)
        absolute_only.add_node(node)
        relative.add_node(node)

    absolute_only.rebalance()
    relative.rebalance()

    assert absolute_only.root.is_leaf
    assert [child.count for child in relative.root.children or []] == [6, 6]


def test_labels_via_injected_labeler():
    tree = build_topical_tree()
    tree.organize(labeler=FunctionLabeler(topic_labeler))
    root = tree.root
    assert root.label in TOPICS  # root labeled with dominant topic
    leaf_labels = {leaf.label for leaf in root.children}
    assert leaf_labels == set(TOPICS)  # each leaf named by its topic


def test_browse_dict_and_show():
    tree = build_topical_tree()
    tree.organize(labeler=FunctionLabeler(topic_labeler))

    branch = tree.to_branch(max_items=2)
    assert branch.count == 30
    leaf = branch.children[0]
    assert isinstance(leaf, BranchNode)
    assert len(leaf.children) <= 2
    assert leaf.label in TOPICS

    text = tree.show(max_items=2)
    assert any(topic in text for topic in TOPICS)
    assert "[30]" in text  # root size shown


def test_single_leaf_branch_is_lazy_labeled_and_collapsible():
    tree = build_topical_tree()
    tree.rebalance()
    tree.remove_batch([*(f"animal:{i}" for i in range(10)), *(f"food:{i}" for i in range(10))])

    calls = []

    def counting_labeler(request: LabelRequest) -> str:
        calls.append(request)
        return "should not be used"

    tree.label(labeler=FunctionLabeler(counting_labeler))
    assert calls == []  # no LLM/labeler call for a non-branching wrapper + sole leaf

    branch = tree.to_branch(max_items=2, collapse_single_leaf=True)
    assert branch.count == 10
    assert branch.label.startswith("language note")
    assert all(isinstance(child, ContentNode) for child in branch.children)

    text = tree.show(max_items=1, collapse_single_leaf=True)
    assert "(unlabeled)" not in text
    assert "language note" in text


def test_organize_persistence_keeps_labels(tmp_path):
    path = os.path.join(tmp_path, "tax.json")
    t1 = EmbedTree(
        embedder=topical_embedder(),
        state=JsonTreeLoader(path),
        config=TreeConfig(max_branches=3, leaf_capacity=10, model_args={"random_state": 0}),
    )
    for topic in TOPICS:
        for i in range(10):
            t1.add_node(ContentNode(f"{topic}:{i}", f"{topic} note {i}"))
    t1.organize(labeler=FunctionLabeler(topic_labeler))
    labels_before = sorted(c.label for c in t1.root.children)

    t2 = EmbedTree(
        embedder=topical_embedder(),
        state=JsonTreeLoader(path),
        config=TreeConfig(max_branches=3, leaf_capacity=10),
    )
    assert len(t2) == 30
    labels_after = sorted(c.label for c in t2.root.children)
    assert labels_before == labels_after  # labels survived reload


def test_keyword_labeler_default_when_no_llm():
    # No labeler injected, provider "none" -> KeywordLabeler names nodes.
    tree = build_topical_tree()
    tree.organize()  # uses config.llm (provider none) -> TF-IDF
    for leaf in tree.root.children:
        assert leaf.label  # non-empty keyword label
