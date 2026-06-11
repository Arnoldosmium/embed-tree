"""JSON-backed tree loader."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from embed_tree.persisters.model import MaterializedTreeState
from embed_tree.representation import PartialTree
from embed_tree.representation.default import partial_tree_from_dict, partial_tree_to_dict


class JsonTreeLoader:
    """Load/save a PartialTree or materialized state as one JSON file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.post_init()

    def post_init(self) -> None:
        """Hook for implementations that need setup after construction."""
        pass

    def load(self) -> PartialTree | MaterializedTreeState | None:
        if not self.path.exists():
            return None
        with self.path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("kind") == "partial_tree":
            return partial_tree_from_dict(data["tree"])
        if data.get("kind") == "materialized_tree_state":
            return data["state"]
        return data

    def save(self, state: PartialTree | MaterializedTreeState) -> None:
        if isinstance(state, PartialTree):
            payload: dict[str, Any] = {"kind": "partial_tree", "tree": partial_tree_to_dict(state)}
        else:
            payload = {"kind": "materialized_tree_state", "state": state}

        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f"{self.path.name}.tmp.{os.getpid()}")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)
