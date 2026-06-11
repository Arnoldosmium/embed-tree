"""File-backed full-snapshot store."""

from __future__ import annotations

import json
import os

from .model import TreeState


class FileTreeStore:
    """Single-file JSON snapshot store with atomic writes."""

    def __init__(self, path: str) -> None:
        self.path = path

    def load(self) -> TreeState | None:
        if not os.path.exists(self.path):
            return None
        with open(self.path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save(self, state: TreeState) -> None:
        parent = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(parent, exist_ok=True)
        tmp = f"{self.path}.tmp.{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)  # atomic on POSIX

