"""LLM-backed streaming labeler."""

from __future__ import annotations

from typing import Any, Iterable

from embed_tree.config import LLMConfig
from embed_tree.taggers import LLMTagger

from .model import LabelRequest


class LLMLabeler:
    """Generate labels from nearby candidates using the existing LLM tagger."""

    def __init__(self, config: LLMConfig | None = None, *, client: Any | None = None, pipeline: Any | None = None) -> None:
        self.config = config or LLMConfig()
        self.tagger = LLMTagger(self.config, client=client, pipeline=pipeline)

    def stream(self, request: LabelRequest) -> Iterable[str]:
        yield self.label(request)

    def label(self, request: LabelRequest) -> str:
        texts = [candidate.text for candidate in request.candidates]
        return self.tagger(texts)

