from __future__ import annotations

import hashlib
import json

import pytest

from embed_tree import (
    BranchNode,
    ContentNode,
    DefaultTreeReconciler,
    EmbedTree,
    FileSystemTreeLoader,
    FolderTreePersister,
    JsonTreeLoader,
    MissingNodeFileError,
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
    leaves = _leaves(tree)
    assert {node.id for node in leaves} == {
        hashlib.md5(b"hello").hexdigest(),
        hashlib.md5(b"install").hexdigest(),
    }
    by_name = {node.metadata["filename"]: node for node in leaves}
    assert by_name["intro.md"].metadata["path"] == str(root / "intro.md")
    assert by_name["intro.md"].metadata["relative_path"] == "intro.md"
    assert by_name["setup.md"].metadata["path"] == str(root / "guides" / "setup.md")
    assert by_name["setup.md"].metadata["relative_path"] == "guides/setup.md"


def test_filesystem_loader_accepts_text_generator(tmp_path):
    root = tmp_path / "docs"
    root.mkdir()
    (root / "intro.md").write_text("# Intro\n\nbody text", encoding="utf-8")

    tree = FileSystemTreeLoader(
        root,
        include_suffixes=[".md"],
        text_generator=lambda path, raw: raw.splitlines()[0].lstrip("# "),
    ).load()

    assert tree is not None
    leaf = _leaves(tree)[0]
    assert leaf.id == hashlib.md5(b"# Intro\n\nbody text").hexdigest()
    assert leaf.text == "Intro"
    assert leaf.metadata["filename"] == "intro.md"


def test_filesystem_loader_accepts_additional_metadata_derivers(tmp_path):
    root = tmp_path / "docs"
    root.mkdir()
    (root / "intro.md").write_text("title: Getting Started\n\nbody", encoding="utf-8")

    tree = FileSystemTreeLoader(
        root,
        include_suffixes=[".md"],
        additional_metadata_derivers=[
            lambda raw: {
                "new_file_name": "draft.md",
                "title": raw.splitlines()[0].removeprefix("title: "),
            },
            lambda raw: {
                "new_file_name": raw.splitlines()[0].removeprefix("title: ").lower().replace(" ", "-") + ".md",
                "tags": ["docs"],
            },
        ],
    ).load()

    assert tree is not None
    leaf = _leaves(tree)[0]
    assert leaf.id == hashlib.md5(b"title: Getting Started\n\nbody").hexdigest()
    assert leaf.text == "title: Getting Started\n\nbody"
    assert leaf.metadata["filename"] == "intro.md"
    assert leaf.metadata["new_file_name"] == "getting-started.md"
    assert leaf.metadata["title"] == "Getting Started"
    assert leaf.metadata["tags"] == ["docs"]


def test_filesystem_loader_rejects_single_additional_metadata_deriver(tmp_path):
    with pytest.raises(TypeError, match="additional_metadata_derivers"):
        FileSystemTreeLoader(
            tmp_path,
            additional_metadata_derivers=lambda raw: {"title": raw},
        )


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

    FolderTreePersister(root, include_suffixes=[".md"], missing_node_file="create").save(tree)

    alpha = json.loads((root / "Topic One" / "alpha.txt").read_text(encoding="utf-8"))
    beta = json.loads((root / "Topic One" / "beta.txt").read_text(encoding="utf-8"))
    assert alpha == {
        "text": "alpha",
        "metadata": {"filename": "alpha.md", "content": "alpha"},
    }
    assert beta == {
        "text": "beta",
        "metadata": {"filename": "beta.md", "content": "beta"},
    }


def test_folder_tree_persister_moves_by_md5_and_can_rename(tmp_path):
    root = tmp_path / "docs"
    (root / "old").mkdir(parents=True)
    (root / "old" / "alpha.md").write_text("alpha", encoding="utf-8")
    alpha_id = hashlib.md5(b"alpha").hexdigest()
    tree = BranchNode(
        id="root",
        children=[
            BranchNode(
                id="topic",
                label="New Topic",
                children=[
                    ContentNode(
                        id=alpha_id,
                        text="alpha",
                        metadata={
                            "relative_path": "old/alpha.md",
                            "filename": "alpha.md",
                            "new_file_name": "renamed.md",
                        },
                    )
                ],
            )
        ],
    )

    FolderTreePersister(root, include_suffixes=[".md"], missing_node_file="create").save(tree)

    assert not (root / "old" / "alpha.md").exists()
    assert (root / "New Topic" / "renamed.md").read_text(encoding="utf-8") == "alpha"


def test_folder_tree_persister_copies_from_metadata_path_when_md5_matches(tmp_path):
    root = tmp_path / "docs"
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source = source_dir / "alpha.md"
    source.write_text("alpha", encoding="utf-8")
    alpha_id = hashlib.md5(b"alpha").hexdigest()
    tree = BranchNode(
        id="root",
        children=[
            BranchNode(
                id="topic",
                label="Copied Topic",
                children=[
                    ContentNode(
                        id=alpha_id,
                        text="summary",
                        metadata={
                            "path": str(source),
                            "new_file_name": "copied.md",
                        },
                    )
                ],
            )
        ],
    )

    FolderTreePersister(root, include_suffixes=[".md"], missing_node_file="create").save(tree)

    assert source.read_text(encoding="utf-8") == "alpha"
    assert (root / "Copied Topic" / "copied.md").read_text(encoding="utf-8") == "alpha"


def test_folder_tree_persister_snapshots_when_metadata_path_md5_differs(tmp_path):
    root = tmp_path / "docs"
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source = source_dir / "alpha.md"
    source.write_text("different", encoding="utf-8")
    alpha_id = hashlib.md5(b"alpha").hexdigest()
    tree = BranchNode(
        id="root",
        children=[
            ContentNode(
                id=alpha_id,
                text="summary",
                metadata={
                    "path": str(source),
                    "filename": "alpha.md",
                },
            )
        ],
    )

    FolderTreePersister(root, include_suffixes=[".md"], missing_node_file="create").save(tree)

    assert source.read_text(encoding="utf-8") == "different"
    snapshot = json.loads((root / "alpha.txt").read_text(encoding="utf-8"))
    assert snapshot == {
        "text": "summary",
        "metadata": {
            "path": str(source),
            "filename": "alpha.md",
        },
    }


def test_folder_tree_persister_writes_generic_node_as_new_file(tmp_path):
    root = tmp_path / "docs"
    root.mkdir()
    (root / "alpha.md").write_text("alpha", encoding="utf-8")
    tree = BranchNode(
        id="root",
        children=[
            ContentNode(
                id="generic-alpha",
                text="alpha",
                metadata={
                    "relative_path": "alpha.md",
                    "filename": "alpha.md",
                },
            )
        ],
    )

    FolderTreePersister(root, include_suffixes=[".md"], missing_node_file="create").save(tree)

    assert (root / "alpha.md").read_text(encoding="utf-8") == "alpha"
    snapshot = json.loads((root / "alpha.txt").read_text(encoding="utf-8"))
    assert snapshot == {
        "text": "alpha",
        "metadata": {
            "relative_path": "alpha.md",
            "filename": "alpha.md",
        },
    }


def test_folder_tree_persister_skips_missing_node_file_by_default(tmp_path):
    root = tmp_path / "docs"
    tree = BranchNode(id="root", children=[ContentNode(id="generic-alpha", text="alpha", metadata={"filename": "alpha.md"})])

    with pytest.warns(RuntimeWarning, match="skipping"):
        FolderTreePersister(root, include_suffixes=[".md"]).save(tree)

    assert not (root / "alpha.txt").exists()


def test_folder_tree_persister_raises_for_missing_node_file(tmp_path):
    root = tmp_path / "docs"
    tree = BranchNode(id="root", children=[ContentNode(id="generic-alpha", text="alpha", metadata={"filename": "alpha.md"})])

    with pytest.raises(MissingNodeFileError):
        FolderTreePersister(root, include_suffixes=[".md"], missing_node_file="raise").save(tree)

    assert not (root / "alpha.txt").exists()


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
