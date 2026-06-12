"""Keyword labeler."""

from __future__ import annotations

from typing import Iterable

from sklearn.feature_extraction.text import TfidfVectorizer

from .model import LabelRequest


class KeywordLabeler:
    """Generate short labels from TF-IDF terms without network calls."""

    def __init__(self, top_k: int = 3) -> None:
        self.top_k = top_k

    def stream(self, request: LabelRequest) -> Iterable[str]:
        yield self.label(request)

    def label(self, request: LabelRequest) -> str:
        texts = [candidate.text for candidate in request.candidates if candidate.text.strip()]
        if not texts:
            return ""
        try:
            vectorizer = TfidfVectorizer(stop_words="english", max_features=50)
            matrix = vectorizer.fit_transform(texts)
        except ValueError:
            return ""
        scores = matrix.sum(axis=0).A1
        terms = vectorizer.get_feature_names_out()
        order = scores.argsort()[::-1][: self.top_k]
        return ", ".join(str(terms[i]) for i in order if scores[i] > 0)
