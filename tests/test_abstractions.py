from __future__ import annotations

import hashlib

import numpy as np
import pytest

from embed_tree import (
    ContentNode,
    DefaultTreeReconciler,
    EmbedTree,
    FakeEmbeddingProvider,
    FileSystemTreeLoader,
    FolderTreePersister,
    Item,
    JsonTreeLoader,
    KeyNode,
    Node,
    NodeEmbedding,
    PartialTree,
    SQLAlchemyContentLoader,
    SQLAlchemyTreePersister,
    SQLiteTreeLoader,
    TreeEdge,
    TreeConfig,
)


def test_filesystem_loader_builds_partial_tree(tmp_path):
    root = tmp_path / "docs"
    (root / "guides").mkdir(parents=True)
    (root / "intro.md").write_text("hello", encoding="utf-8")
    (root / "guides" / "setup.md").write_text("install", encoding="utf-8")
    (root / "skip.bin").write_bytes(b"\xff")

    tree = FileSystemTreeLoader(root, include_suffixes=[".md"]).load()

    assert tree is not None
    intro_id = hashlib.md5(b"hello").hexdigest()
    setup_id = hashlib.md5(b"install").hexdigest()
    assert {n.id for n in tree.content_nodes} == {intro_id, setup_id}
    assert {n.id for n in tree.key_nodes} >= {".", "guides"}
    assert any(e.parent_id == "guides" and e.child_id == setup_id for e in tree.edges)


def test_embed_tree_adds_loader_content_nodes(tmp_path):
    root = tmp_path / "docs"
    root.mkdir()
    (root / "intro.md").write_text("hello", encoding="utf-8")
    loaded = FileSystemTreeLoader(root, include_suffixes=[".md"]).load()
    assert loaded is not None
    intro_id = hashlib.md5(b"hello").hexdigest()
    tree = EmbedTree(
        embedder=FakeEmbeddingProvider(dim=16),
        config=TreeConfig(leaf_capacity=10, max_branches=2),
    )

    ids = tree.add_partial_tree(loaded)

    assert ids == [intro_id]
    assert tree.query("hello", k=1, exhaustive=True)[0][0] == intro_id


def test_folder_tree_persister_materializes_known_files_and_ignores_unknown(tmp_path):
    root = tmp_path / "docs"
    (root / "old").mkdir(parents=True)
    (root / "old" / "alpha.md").write_text("alpha", encoding="utf-8")
    (root / "old" / "unknown.md").write_text("unknown", encoding="utf-8")
    alpha_id = hashlib.md5(b"alpha").hexdigest()
    beta_id = hashlib.md5(b"beta").hexdigest()

    tree = PartialTree(
        key_nodes=[
            KeyNode(id=".", label="docs"),
            KeyNode(id="topic", label="Topic One"),
        ],
        content_nodes=[
            ContentNode(id=alpha_id, content=None),
            ContentNode(id=beta_id, content="beta", payload={"filename": "beta.md"}),
        ],
        edges=[
            TreeEdge(parent_id=".", child_id="topic"),
            TreeEdge(parent_id="topic", child_id=alpha_id),
            TreeEdge(parent_id="topic", child_id=beta_id),
        ],
    )

    FolderTreePersister(root, include_suffixes=[".md"]).save(tree)

    assert not (root / "old" / "alpha.md").exists()
    assert (root / "Topic One" / "alpha.md").read_text(encoding="utf-8") == "alpha"
    assert (root / "Topic One" / "beta.md").read_text(encoding="utf-8") == "beta"
    assert (root / "old" / "unknown.md").read_text(encoding="utf-8") == "unknown"


def test_folder_tree_persister_prunes_empty_folders_after_move(tmp_path):
    root = tmp_path / "docs"
    (root / "old").mkdir(parents=True)
    (root / "old" / "alpha.md").write_text("alpha", encoding="utf-8")
    alpha_id = hashlib.md5(b"alpha").hexdigest()
    tree = PartialTree(
        key_nodes=[KeyNode(id=".", label="docs"), KeyNode(id="topic", label="Topic")],
        content_nodes=[ContentNode(id=alpha_id, content=None)],
        edges=[TreeEdge(".", "topic"), TreeEdge("topic", alpha_id)],
    )

    FolderTreePersister(root, include_suffixes=[".md"]).save(tree)

    assert not (root / "old").exists()
    assert (root / "Topic" / "alpha.md").is_file()


def test_folder_tree_persister_materializes_live_tree(tmp_path):
    root = tmp_path / "docs"
    (root / "uncategorized").mkdir(parents=True)
    (root / "uncategorized" / "alpha.md").write_text("alpha", encoding="utf-8")
    alpha_id = hashlib.md5(b"alpha").hexdigest()
    item = Item(
        id=alpha_id,
        vector=np.asarray([1.0, 0.0]),
        payload={"filename": "alpha.md"},
        text="alpha",
    )
    leaf = Node(
        id=1,
        vsum=np.asarray([1.0, 0.0]),
        count=1,
        items=[item],
        label="Live Topic",
    )
    root_node = Node(id=0, vsum=np.asarray([1.0, 0.0]), count=1, children=[leaf])

    FolderTreePersister(root, include_suffixes=[".md"]).save(root_node)

    assert not (root / "uncategorized").exists()
    assert (root / "Live Topic" / "alpha.md").read_text(encoding="utf-8") == "alpha"


def test_json_tree_loader_persists_round_trip(tmp_path):
    path = tmp_path / "tree_state.json"
    persister = JsonTreeLoader(path)
    tree = PartialTree(
        content_nodes=[ContentNode(id="a", content="hello", text="A")],
        embeddings={"a": NodeEmbedding(node_id="a", vector=[1.0, 0.0])},
        labels={"a": "alpha"},
    )

    persister.save(tree)
    loaded = persister.load()

    assert isinstance(loaded, PartialTree)
    assert loaded.content_nodes[0].id == "a"
    assert loaded.embeddings["a"].vector == [1.0, 0.0]
    assert loaded.labels["a"] == "alpha"


def test_default_reconciler_reuses_loaded_state_and_embeds_new_items():
    ground_truth = PartialTree(
        content_nodes=[
            ContentNode(id="keep", content="cached"),
            ContentNode(id="new", content="fresh"),
        ]
    )
    reusable_state = PartialTree(
        content_nodes=[ContentNode(id="keep", content="cached"), ContentNode(id="stale", content="old")],
        embeddings={
            "keep": NodeEmbedding(node_id="keep", vector=[1.0, 0.0]),
            "stale": NodeEmbedding(node_id="stale", vector=[0.0, 1.0]),
        },
        labels={"keep": "cached label", "stale": "old label"},
    )

    reconciled = DefaultTreeReconciler().reconcile(
        StaticTreeLoader(ground_truth),
        StaticTreeLoader(reusable_state),
        embedder=lambda content: [0.5, 0.5],
    )

    assert isinstance(reconciled, PartialTree)
    assert set(reconciled.embeddings) == {"keep", "new"}
    assert reconciled.embeddings["keep"].vector == [1.0, 0.0]
    assert reconciled.embeddings["new"].vector == [0.5, 0.5]
    assert reconciled.labels == {"keep": "cached label"}


def test_sqlite_tree_loader_persists_round_trip(tmp_path):
    sa = pytest.importorskip("sqlalchemy")
    url = f"sqlite:///{tmp_path / 'tree.db'}"
    persister = SQLiteTreeLoader(tmp_path / "tree_state.db")
    tree = PartialTree(content_nodes=[ContentNode(id="a", content="hello")])

    persister.save(tree)
    loaded = persister.load()

    assert isinstance(loaded, PartialTree)
    assert loaded.content_nodes[0].id == "a"

    persister = SQLAlchemyTreePersister(url, "source_docs")
    persister.save(tree)
    loaded_source = SQLAlchemyContentLoader(sa.create_engine(url), "source_docs").load()

    assert loaded_source is not None
    assert loaded_source.content_nodes[0].id == "a"
    assert loaded_source.content_nodes[0].content == "hello"


class StaticTreeLoader:
    def __init__(self, tree):
        self.tree = tree

    def load(self):
        return self.tree
