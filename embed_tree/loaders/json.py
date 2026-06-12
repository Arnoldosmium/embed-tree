"""JSON-backed tree/state loader."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from embed_tree.persisters.model import MaterializedTreeState
from embed_tree.representation import BranchNode
from embed_tree.representation.default import tree_from_dict, tree_to_dict


class JsonTreeLoader:
    """Load/save a BranchNode or materialized EmbedTree state as JSON."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.post_init()

    def post_init(self) -> None:
        pass

    def load(self) -> BranchNode | MaterializedTreeState | None:
        if not self.path.exists():
            return None
        with self.path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("kind") == "branch":
            tree = tree_from_dict(data["tree"])
            if not isinstance(tree, BranchNode):
                raise ValueError("JSON root must be a branch")
            return tree
        if data.get("kind") == "materialized_tree_state":
            return data["state"]
        return data

    def save(self, state: BranchNode | MaterializedTreeState) -> None:
        if isinstance(state, BranchNode):
            payload: dict[str, Any] = {"kind": "branch", "tree": tree_to_dict(state)}
        else:
            payload = {"kind": "materialized_tree_state", "state": state}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f"{self.path.name}.tmp.{os.getpid()}")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)
