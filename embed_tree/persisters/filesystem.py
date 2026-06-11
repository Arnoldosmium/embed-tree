"""Folder-backed external persister."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from embed_tree.representation import ContentNode, KeyNode, NodeId, PartialTree

_UNSAFE_PATH_CHARS = re.compile(r'[\\/:*?"<>|#\[\]^\n\r\t]+')


class FolderTreePersister:
    """Materialize a live tree as folders and files.

    The local folder is treated as mutable ground truth: every save reloads its
    current file-md5 map, compares it with the target layout from the in-memory
    tree, moves known files into place, creates missing files only when content
    is available, prunes empty folders, and leaves unknown files untouched.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        include_suffixes: set[str] | list[str] | tuple[str, ...] | None = None,
        encoding: str = "utf-8",
        include_hidden: bool = False,
    ) -> None:
        self.root = Path(root)
        self.include_suffixes = None if include_suffixes is None else {s.lower() for s in include_suffixes}
        self.encoding = encoding
        self.include_hidden = include_hidden

    def save(self, state: Any) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        current = self._current_files_by_md5()
        desired = self._desired_files(state, current)

        reserved: set[Path] = set()
        for file in desired:
            target = _unique_path(self.root / file.folder / file.filename, reserved)
            reserved.add(target)
            existing = current.get(file.md5)
            if existing is not None:
                self._move_file(existing, target)
            elif file.content is not None:
                self._write_file(target, file.content)

        self._prune_empty_dirs()

    def _desired_files(self, state: Any, current: dict[str, Path]) -> list["_DesiredFile"]:
        tree = state.get_tree() if hasattr(state, "get_tree") and callable(state.get_tree) else state
        if isinstance(tree, PartialTree):
            return self._desired_files_from_partial_tree(tree, current)
        if _looks_like_live_node(tree):
            return list(self._desired_files_from_live_node(tree, current))
        raise TypeError("FolderTreePersister persists an EmbedTree, live Node, or PartialTree")

    def _current_files_by_md5(self) -> dict[str, Path]:
        files: dict[str, Path] = {}
        for path in sorted(self.root.rglob("*")):
            if not path.is_file() or not self._included(path):
                continue
            rel_parts = path.relative_to(self.root).parts
            if not self.include_hidden and any(part.startswith(".") for part in rel_parts):
                continue
            files.setdefault(_file_md5(path), path)
        return files

    def _desired_files_from_partial_tree(
        self,
        tree: PartialTree,
        current: dict[str, Path],
    ) -> list["_DesiredFile"]:
        key_nodes = {node.id: node for node in tree.key_nodes}
        parent_by_child = {edge.child_id: edge.parent_id for edge in tree.edges}
        desired: list[_DesiredFile] = []

        for node in tree.content_nodes:
            folder = self._folder_for(node.id, key_nodes, parent_by_child)
            filename = self._filename_for_node(node, current.get(str(node.id)))
            content = None if node.content is None else str(node.content)
            desired.append(_DesiredFile(str(node.id), folder, filename, content))
        return desired

    def _desired_files_from_live_node(
        self,
        root: Any,
        current: dict[str, Path],
    ) -> list["_DesiredFile"]:
        if root.is_leaf:
            folder = Path(_safe_name(root.label or "topics"))
            return [
                file
                for item in root.items or []
                if (file := self._desired_file_for_item(item, folder, current)) is not None
            ]

        desired: list[_DesiredFile] = []
        self._collect_live_node_files(root, Path(), current, desired, include_self=False)
        return desired

    def _collect_live_node_files(
        self,
        node: Any,
        prefix: Path,
        current: dict[str, Path],
        desired: list["_DesiredFile"],
        *,
        include_self: bool,
    ) -> None:
        folder = prefix
        if include_self:
            folder = prefix / _safe_name(node.label or f"topic-{getattr(node, 'id', 'node')}")
        if node.is_leaf:
            for item in node.items or []:
                file = self._desired_file_for_item(item, folder, current)
                if file is not None:
                    desired.append(file)
            return

        used: set[str] = set()
        for index, child in enumerate(node.children or []):
            label = _dedupe_name(_safe_name(child.label or f"topic-{index + 1}"), used)
            self._collect_live_node_files(child, folder / label, current, desired, include_self=False)

    def _desired_file_for_item(
        self,
        item: Any,
        folder: Path,
        current: dict[str, Path],
    ) -> "_DesiredFile | None":
        md5 = self._md5_for_item(item)
        if md5 is None:
            return None
        existing = current.get(md5)
        filename = self._filename_for_item(item, existing)
        content = self._content_for_item(item)
        return _DesiredFile(md5, folder, filename, content)

    def _folder_for(
        self,
        node_id: NodeId,
        key_nodes: dict[NodeId, KeyNode],
        parent_by_child: dict[NodeId, NodeId],
    ) -> Path:
        parts: list[str] = []
        current = parent_by_child.get(node_id)
        seen: set[NodeId] = set()
        while current is not None and current not in seen:
            seen.add(current)
            if str(current) == ".":
                break
            key = key_nodes.get(current)
            raw = key.label if key is not None and key.label else str(current)
            part = _safe_name(raw)
            if part:
                parts.append(part)
            current = parent_by_child.get(current)
        return Path(*reversed(parts)) if parts else Path()

    def _filename_for_node(self, node: ContentNode, existing: Path | None) -> str:
        if existing is not None:
            return existing.name
        payload = node.payload if isinstance(node.payload, dict) else {}
        for key in ("filename", "relative_path", "path"):
            value = payload.get(key)
            if value:
                name = Path(str(value)).name
                if name:
                    return _safe_name(name, fallback=f"{node.id}.txt")
        text = node.text.strip() if isinstance(node.text, str) else ""
        return _safe_name(text, fallback=f"{node.id}.txt")

    def _md5_for_item(self, item: Any) -> str | None:
        item_id = str(item.id)
        if _is_md5(item_id):
            return item_id
        payload = item.payload if isinstance(item.payload, dict) else {}
        for key in ("md5", "file_md5", "content_md5", "content_id", "id"):
            value = payload.get(key)
            if value is not None and _is_md5(str(value)):
                return str(value)
        return None

    def _filename_for_item(self, item: Any, existing: Path | None) -> str:
        if existing is not None:
            return existing.name
        payload = item.payload if isinstance(item.payload, dict) else {}
        for key in ("filename", "relative_path", "output_path", "path"):
            value = payload.get(key)
            if value:
                name = Path(str(value)).name
                if name:
                    return _safe_name(name, fallback=f"{item.id}.txt")
        text = item.text.strip() if isinstance(item.text, str) else ""
        return _safe_name(text, fallback=f"{item.id}.txt")

    def _content_for_item(self, item: Any) -> str | None:
        payload = item.payload if isinstance(item.payload, dict) else {}
        for key in ("content", "body", "text"):
            value = payload.get(key)
            if value is not None:
                return str(value)
        return None

    def _move_file(self, source: Path, target: Path) -> None:
        if source == target:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        source.rename(target)

    def _write_file(self, target: Path, content: str) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding=self.encoding)

    def _prune_empty_dirs(self) -> None:
        for directory in sorted(
            (path for path in self.root.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            rel_parts = directory.relative_to(self.root).parts
            if not self.include_hidden and any(part.startswith(".") for part in rel_parts):
                continue
            try:
                next(directory.iterdir())
            except StopIteration:
                directory.rmdir()

    def _included(self, path: Path) -> bool:
        return self.include_suffixes is None or path.suffix.lower() in self.include_suffixes


FileSystemTreePersister = FolderTreePersister


@dataclass(frozen=True)
class _DesiredFile:
    md5: str
    folder: Path
    filename: str
    content: str | None


def _file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_name(value: str, *, fallback: str = "untitled") -> str:
    cleaned = _UNSAFE_PATH_CHARS.sub(" ", value)
    cleaned = " ".join(cleaned.split()).strip(". ")
    return cleaned or fallback


def _dedupe_name(value: str, used: set[str]) -> str:
    candidate = value
    i = 2
    while candidate in used:
        candidate = f"{value}-{i}"
        i += 1
    used.add(candidate)
    return candidate


def _is_md5(value: str) -> bool:
    return len(value) == 32 and all(c in "0123456789abcdefABCDEF" for c in value)


def _looks_like_live_node(value: Any) -> bool:
    return hasattr(value, "is_leaf") and (hasattr(value, "items") or hasattr(value, "children"))


def _unique_path(target: Path, reserved: set[Path]) -> Path:
    if target not in reserved and not target.exists():
        return target
    stem = target.stem
    suffix = target.suffix
    parent = target.parent
    i = 2
    while True:
        candidate = parent / f"{stem}-{i}{suffix}"
        if candidate not in reserved and not candidate.exists():
            return candidate
        i += 1
