"""Filesystem-backed tree loader."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable, Iterable

from embed_tree.representation import BranchNode, ContentNode


class FileSystemTreeLoader:
    """Load files under a directory as a recursive BranchNode tree.

    File ContentNode.id is the file content MD5. Path fields are metadata only;
    they are used by persisters to plan moves when materializing a new tree. A
    text_generator can derive embed text from raw file text without changing
    file identity.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        include_suffixes: Iterable[str] | None = None,
        encoding: str = "utf-8",
        include_hidden: bool = False,
        text_generator: Callable[[Path, str], str] | None = None,
    ) -> None:
        self.root = Path(root)
        self.include_suffixes = None if include_suffixes is None else {s.lower() for s in include_suffixes}
        self.encoding = encoding
        self.include_hidden = include_hidden
        self.text_generator = text_generator

    def load(self) -> BranchNode | None:
        if not self.root.exists():
            return None
        return self._load_dir(self.root, ".")

    def _load_dir(self, path: Path, node_id: str) -> BranchNode:
        branch = BranchNode(
            id=node_id,
            label=path.name or str(path),
            metadata={"path": str(path), "relative_path": "." if path == self.root else path.relative_to(self.root).as_posix()},
        )
        for child in sorted(path.iterdir()):
            if not self.include_hidden and child.name.startswith("."):
                continue
            rel = child.relative_to(self.root).as_posix()
            if child.is_dir():
                branch.children.append(self._load_dir(child, rel))
                continue
            if not child.is_file() or not self._included(child):
                continue
            try:
                raw_text = child.read_text(encoding=self.encoding)
            except UnicodeDecodeError:
                continue
            text = self._text_for_file(child, raw_text)
            file_id = _file_md5(child)
            branch.children.append(
                ContentNode(
                    id=file_id,
                    text=text,
                    metadata={
                        "path": str(child),
                        "relative_path": rel,
                        "filename": child.name,
                        "version": file_id,
                    },
                )
            )
        return branch

    def _included(self, path: Path) -> bool:
        return self.include_suffixes is None or path.suffix.lower() in self.include_suffixes

    def _text_for_file(self, path: Path, raw_text: str) -> str:
        if self.text_generator is None:
            return raw_text
        text = self.text_generator(path, raw_text)
        if not isinstance(text, str):
            raise TypeError("FileSystemTreeLoader text_generator must return str")
        return text


def _file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
