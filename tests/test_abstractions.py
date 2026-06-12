from __future__ import annotations

import hashlib

from embed_tree import (
    BranchNode,
    ContentNode,
    DefaultTreeReconciler,
    EmbedTree,
    FileSystemTreeLoader,
    FolderTreePersister,
    JsonTreeLoader,
    SQLAlchemyContentLoader,
    SQLAlchemyTreePersister,
    SQLiteTreeLoader,
    TreeConfig,
)
from tests.helpers import FakeTextEmbedder


def test_filesystem_loader_builds_branch_tree(tmp_path):
    root = tmp_path / "docs"
    (root / "guides").mkdir(parents=True)
    (root / "intro.md").write_text("hello", encoding="utf-8")
    (root / "guides" / "setup.md").write_text("install", encoding="utf-8")

    tree = FileSystemTreeLoader(root, include_suffixes=[".md"]).load()

    assert isinstance(tree, BranchNode)
    assert tree.label == "docs"
    assert tree.count == 2
    assert {node.id for node in _leaves(tree)} == {
        hashlib.md5(b"hello").hexdigest(),
        hashlib.md5(b"install").hexdigest(),
    }


def test_embed_tree_adds_branch(tmp_path):
    root = tmp_path / "docs"
    root.mkdir()
    (root / "intro.md").write_text("hello", encoding="utf-8")
    loaded = FileSystemTreeLoader(root, include_suffixes=[".md"]).load()
    assert loaded is not None

    tree = EmbedTree(FakeTextEmbedder(dim=16), config=TreeConfig(leaf_capacity=10, max_branches=2))
    ids = tree.add_branch(loaded)

    intro_id = hashlib.md5(b"hello").hexdigest()
    assert ids == [intro_id]
    assert tree.query("hello", k=1, exhaustive=True)[0][0] == intro_id


def test_folder_tree_persister_materializes_branch(tmp_path):
    root = tmp_path / "docs"
    tree = BranchNode(
        id="root",
        label="Docs",
        children=[
            BranchNode(
                id="topic",
                label="Topic One",
                children=[
                    ContentNode("a", "alpha", {"filename": "alpha.md", "content": "alpha"}),
                    ContentNode("b", "beta", {"filename": "beta.md", "content": "beta"}),
                ],
            )
        ],
    )

    FolderTreePersister(root, include_suffixes=[".md"]).save(tree)

    assert (root / "Topic One" / "alpha.md").read_text(encoding="utf-8") == "alpha"
    assert (root / "Topic One" / "beta.md").read_text(encoding="utf-8") == "beta"


def test_json_tree_loader_persists_branch_round_trip(tmp_path):
    path = tmp_path / "tree.json"
    loader = JsonTreeLoader(path)
    tree = BranchNode(id="root", children=[ContentNode(id="a", text="hello", metadata={"k": "v"})])

    loader.save(tree)
    loaded = loader.load()

    assert isinstance(loaded, BranchNode)
    assert loaded.children[0].id == "a"
    assert loaded.children[0].metadata == {"k": "v"}


def test_default_reconciler_returns_ground_truth():
    tree = BranchNode(id="root", children=[ContentNode(id="a", text="hello")])
    reconciled = DefaultTreeReconciler().reconcile(StaticTreeLoader(tree), embedder=lambda text: [1.0])

    assert reconciled is tree


def test_sqlite_tree_loader_persists_round_trip(tmp_path):
    sa = __import__("pytest").importorskip("sqlalchemy")
    url = f"sqlite:///{tmp_path / 'tree.db'}"
    tree = BranchNode(id="root", children=[ContentNode(id="a", text="hello")])

    loader = SQLiteTreeLoader(tmp_path / "tree_state.db")
    loader.save(tree)
    loaded = loader.load()
    assert isinstance(loaded, BranchNode)
    assert loaded.children[0].id == "a"

    SQLAlchemyTreePersister(url, "source_docs").save(tree)
    loaded_source = SQLAlchemyContentLoader(sa.create_engine(url), "source_docs").load()
    assert loaded_source is not None
    assert loaded_source.children[0].text == "hello"


class StaticTreeLoader:
    def __init__(self, tree):
        self.tree = tree

    def load(self):
        return self.tree


def _leaves(branch: BranchNode) -> list[ContentNode]:
    out: list[ContentNode] = []
    for child in branch.children:
        if isinstance(child, ContentNode):
            out.append(child)
        else:
            out.extend(_leaves(child))
    return out
