"""Filesystem-backed ground-truth loader."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

from embed_tree.representation import ContentNode, KeyNode, PartialTree, TreeEdge


class FileSystemTreeLoader:
    """Load files under a directory as content nodes.

    Directory nodes are emitted as ``KeyNode`` records with edges to their child
    directories/files. File node ids are MD5 hashes of their file bytes, so the
    same file keeps its identity when it moves locally.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        include_suffixes: Iterable[str] | None = None,
        encoding: str = "utf-8",
        include_hidden: bool = False,
    ) -> None:
        self.root = Path(root)
        self.include_suffixes = None if include_suffixes is None else {s.lower() for s in include_suffixes}
        self.encoding = encoding
        self.include_hidden = include_hidden

    def load(self) -> PartialTree | None:
        if not self.root.exists():
            return None

        tree = PartialTree(metadata={"source": "filesystem", "root": str(self.root)})
        root_id = "."
        tree.key_nodes.append(KeyNode(id=root_id, label=self.root.name or str(self.root)))

        for path in sorted(self.root.rglob("*")):
            rel = path.relative_to(self.root).as_posix()
            if not self.include_hidden and any(part.startswith(".") for part in path.relative_to(self.root).parts):
                continue
            parent = path.parent.relative_to(self.root).as_posix() if path.parent != self.root else root_id
            if path.is_dir():
                tree.key_nodes.append(KeyNode(id=rel, label=path.name))
                tree.edges.append(TreeEdge(parent_id=parent, child_id=rel))
                continue
            if not path.is_file() or not self._included(path):
                continue
            file_id = _file_md5(path)
            try:
                content = path.read_text(encoding=self.encoding)
            except UnicodeDecodeError:
                continue
            tree.content_nodes.append(
                ContentNode(
                    id=file_id,
                    content=content,
                    text=path.stem,
                    payload={
                        "path": str(path),
                        "relative_path": rel,
                        "filename": path.name,
                    },
                    version=file_id,
                )
            )
            tree.edges.append(TreeEdge(parent_id=parent, child_id=file_id))

        return tree

    def _included(self, path: Path) -> bool:
        return self.include_suffixes is None or path.suffix.lower() in self.include_suffixes


def _file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
