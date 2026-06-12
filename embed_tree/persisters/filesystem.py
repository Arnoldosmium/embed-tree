"""Folder-backed external persister."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from embed_tree.representation import BranchNode, ContentNode

_UNSAFE_PATH_CHARS = re.compile(r'[\\/:*?"<>|#\[\]^\n\r\t]+')

MissingNodeFileMode = Literal["create", "skip", "raise"]


class MissingNodeFileError(FileNotFoundError):
    """Raised when a content node cannot be matched to a source file."""


class FolderTreePersister:
    """Materialize a BranchNode as folders and files.

    Existing files in root are moved when a node has matching content MD5
    identity. If no current file matches, path metadata may point to a source
    file to copy when its MD5 matches that identity. Otherwise the node is
    skipped by default. Use missing_node_file="create" to write unmatched nodes
    as .txt snapshots containing their text and metadata. Use
    `metadata["new_file_name"]` to rename a moved or copied file.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        include_suffixes: set[str] | list[str] | tuple[str, ...] | None = None,
        encoding: str = "utf-8",
        include_hidden: bool = False,
        missing_node_file: MissingNodeFileMode = "skip",
    ) -> None:
        if missing_node_file not in {"create", "skip", "raise"}:
            raise ValueError("missing_node_file must be 'create', 'skip', or 'raise'")
        self.root = Path(root)
        self.include_suffixes = None if include_suffixes is None else {s.lower() for s in include_suffixes}
        self.encoding = encoding
        self.include_hidden = include_hidden
        self.missing_node_file = missing_node_file

    def save(self, state: BranchNode) -> None:
        if not isinstance(state, BranchNode):
            raise TypeError("FolderTreePersister persists BranchNode instances")
        self.root.mkdir(parents=True, exist_ok=True)
        current = self._current_files_by_md5()
        desired = list(self._desired_files(state, Path(), current, include_self=False))
        reserved: set[Path] = set()
        for file in desired:
            existing = self._existing_file(file, current)
            copy_source = None if existing is not None else self._copy_source(file)
            snapshot = existing is None and copy_source is None
            source = existing or copy_source
            filename = self._filename_for_node(file.node, source, force_txt=snapshot)
            target = _unique_path(self.root / file.folder / filename, reserved, allow_existing=existing)
            reserved.add(target)
            if existing is not None:
                self._move_file(existing, target)
            elif copy_source is not None:
                self._copy_file(copy_source, target)
            else:
                if not self._handle_missing_node_file(file):
                    continue
                self._write_file(target, _generic_node_content(file.node))
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
        return _DesiredFile(
            md5=md5,
            folder=folder,
            node=node,
        )

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

    def _existing_file(self, file: "_DesiredFile", current_by_md5: dict[str, Path]) -> Path | None:
        if file.md5 is None:
            return None
        return current_by_md5.get(file.md5)

    def _copy_source(self, file: "_DesiredFile") -> Path | None:
        if file.md5 is None:
            return None
        source = self._source_path_for_node(file.node)
        if source is None:
            return None
        if not source.is_absolute():
            source = self.root / source
        if not source.is_file():
            return None
        if _file_md5(source) != file.md5:
            return None
        return source

    def _md5_for_node(self, node: ContentNode) -> str | None:
        if _is_md5(str(node.id)):
            return str(node.id)
        for key in ("md5", "file_md5", "content_md5", "content_id"):
            value = node.metadata.get(key)
            if value is not None and _is_md5(str(value)):
                return str(value)
        return None

    def _filename_for_node(self, node: ContentNode, source: Path | None, *, force_txt: bool) -> str:
        new_file_name = node.metadata.get("new_file_name")
        if new_file_name:
            filename = _safe_name(Path(str(new_file_name)).name, fallback=f"{node.id}.txt")
            return _with_suffix(filename, ".txt") if force_txt else filename
        if source is not None:
            return source.name
        for key in ("filename", "relative_path", "output_path", "path"):
            value = node.metadata.get(key)
            if value:
                name = Path(str(value)).name
                if name:
                    filename = _safe_name(name, fallback=f"{node.id}.txt")
                    return _with_suffix(filename, ".txt") if force_txt else filename
        filename = _safe_name(str(node.id), fallback=f"{node.id}.txt")
        return _with_suffix(filename, ".txt") if force_txt else filename

    def _source_path_for_node(self, node: ContentNode) -> Path | None:
        for key in ("path", "relative_path", "source_path"):
            value = node.metadata.get(key)
            if value:
                return Path(str(value))
        return None

    def _move_file(self, source: Path, target: Path) -> None:
        if source == target:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        source.rename(target)

    def _copy_file(self, source: Path, target: Path) -> None:
        if source == target:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    def _write_file(self, target: Path, content: str) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding=self.encoding)

    def _handle_missing_node_file(self, file: "_DesiredFile") -> bool:
        message = f"No matching file found for content node {file.node.id!r}; writing snapshot"
        if self.missing_node_file == "create":
            return True
        if self.missing_node_file == "skip":
            warnings.warn(
                f"No matching file found for content node {file.node.id!r}; skipping",
                RuntimeWarning,
                stacklevel=2,
            )
            return False
        raise MissingNodeFileError(message)

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
    md5: str | None
    folder: Path
    node: ContentNode


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


def _with_suffix(filename: str, suffix: str) -> str:
    path = Path(filename)
    return f"{path.stem}{suffix}"


def _generic_node_content(node: ContentNode) -> str:
    return json.dumps(
        {
            "text": node.text,
            "metadata": node.metadata,
        },
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        default=str,
    )


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


def _unique_path(target: Path, reserved: set[Path], *, allow_existing: Path | None = None) -> Path:
    if allow_existing is not None and target == allow_existing and target not in reserved:
        return target
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
