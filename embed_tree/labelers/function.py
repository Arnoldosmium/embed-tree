"""Function-backed labeler."""

from __future__ import annotations

from typing import Callable, Iterable

from .model import LabelRequest


class FunctionLabeler:
    """Adapt a cheap user function into the streaming labeler protocol."""

    def __init__(self, fn: Callable[[LabelRequest], str | Iterable[str]]) -> None:
        self.fn = fn

    def stream(self, request: LabelRequest) -> Iterable[str]:
        out = self.fn(request)
        if isinstance(out, str):
            yield out
        else:
            yield from out

    def label(self, request: LabelRequest) -> str:
        return "".join(self.stream(request)).strip()

