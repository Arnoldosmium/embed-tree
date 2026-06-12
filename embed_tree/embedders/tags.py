"""Zero-cost tag-set embeddings."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from .model import Vector


class TagSetEmbedder:
    """Embed tags as a fixed-order multi-hot vector.

    Input can be:
    - an iterable of tag strings;
    - a mapping with a `tags` field;
    - a single tag string.
    """

    def __init__(self, tags: Iterable[str], *, unknown: str = "ignore") -> None:
        if unknown not in {"ignore", "error"}:
            raise ValueError("unknown must be 'ignore' or 'error'")
        self.tags = list(dict.fromkeys(tags))
        self.index = {tag: i for i, tag in enumerate(self.tags)}
        self.unknown = unknown

    def embed(self, content: Any) -> Vector:
        vec = np.zeros(len(self.tags), dtype=np.float32)
        for tag in self._tags_from(content):
            i = self.index.get(tag)
            if i is None:
                if self.unknown == "error":
                    raise ValueError(f"unknown tag: {tag!r}")
                continue
            vec[i] = 1.0
        return vec

    def embed_batch(self, contents: Iterable[Any]) -> np.ndarray:
        return np.stack([self.embed(content) for content in contents])

    def __call__(self, content: Any) -> Vector:
        return self.embed(content)

    def _tags_from(self, content: Any) -> list[str]:
        if isinstance(content, str):
            return [content]
        if isinstance(content, Mapping):
            content = content.get("tags", [])
        return [str(tag) for tag in content]
