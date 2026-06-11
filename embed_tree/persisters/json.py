"""JSON external persister."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from embed_tree.representation import PartialTree
from embed_tree.representation.default import partial_tree_to_dict


class JsonTreePersister:
    """Persist an exported tree artifact to JSON."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def save(self, state: Any) -> None:
        payload = partial_tree_to_dict(state) if isinstance(state, PartialTree) else state
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f"{self.path.name}.tmp.{os.getpid()}")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)

