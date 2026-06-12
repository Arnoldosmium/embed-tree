"""Folder-backed external persister."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from embed_tree.representation import BranchNode, ContentNode

_UNSAFE_PATH_CHARS = re.compile(r'[\\/:*?"<>|#\[\]^\n\r\t]+')


class FolderTreePersister:
    """Materialize a BranchNode as folders and files."""

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

    def save(self, state: BranchNode) -> None:
        if not isinstance(state, BranchNode):
            raise TypeError("FolderTreePersister persists BranchNode instances")
        self.root.mkdir(parents=True, exist_ok=True)
        current = self._current_files_by_md5()
        desired = list(self._desired_files(state, Path(), current, include_self=False))
        reserved: set[Path] = set()
        for file in desired:
            target = _unique_path(self.root / file.folder / file.filename, reserved)
            reserved.add(target)
            existing = current.get(file.md5)
            if existing is not None:
                self._move_file(existing, target)
            else:
                self._write_file(target, file.content)
        self._prune_empty_dirs()

    def _desired_files(
        self,
        node: BranchNode | ContentNode,
        folder: Path,
        current: dict[str, Path],
        *,
        include_self: bool,
    ):
        if isinstance(node, ContentNode):
            yield self._desired_file_for_node(node, folder, current)
            return

        next_folder = folder
        if include_self:
            next_folder = folder / _safe_name(node.label or str(node.id))
        used: set[str] = set()
        for index, child in enumerate(node.children):
            if isinstance(child, BranchNode):
                label = _dedupe_name(_safe_name(child.label or f"topic-{index + 1}"), used)
                yield from self._desired_files(child, next_folder / label, current, include_self=False)
            else:
                yield from self._desired_files(child, next_folder, current, include_self=False)

    def _desired_file_for_node(self, node: ContentNode, folder: Path, current: dict[str, Path]) -> "_DesiredFile":
        md5 = self._md5_for_node(node)
        existing = current.get(md5)
        filename = self._filename_for_node(node, existing)
        content = str(node.metadata.get("content", node.text))
        return _DesiredFile(md5, folder, filename, content)

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

    def _md5_for_node(self, node: ContentNode) -> str:
        if _is_md5(str(node.id)):
            return str(node.id)
        for key in ("md5", "file_md5", "content_md5", "content_id"):
            value = node.metadata.get(key)
            if value is not None and _is_md5(str(value)):
                return str(value)
        return hashlib.md5(node.text.encode(self.encoding)).hexdigest()

    def _filename_for_node(self, node: ContentNode, existing: Path | None) -> str:
        if existing is not None:
            return existing.name
        for key in ("filename", "relative_path", "output_path", "path"):
            value = node.metadata.get(key)
            if value:
                name = Path(str(value)).name
                if name:
                    return _safe_name(name, fallback=f"{node.id}.txt")
        return _safe_name(node.text, fallback=f"{node.id}.txt")

    def _move_file(self, source: Path, target: Path) -> None:
        if source == target:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        source.rename(target)

    def _write_file(self, target: Path, content: str) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding=self.encoding)

    def _prune_empty_dirs(self) -> None:
        for directory in sorted((p for p in self.root.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
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
    content: str


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
