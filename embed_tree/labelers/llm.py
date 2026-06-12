"""LLM-backed streaming labeler."""

from __future__ import annotations

from typing import Any, Iterable

from embed_tree.config import LLMConfig

from .model import LabelRequest


class LLMLabeler:
    """Generate labels from nearby candidates using an LLM."""

    def __init__(self, config: LLMConfig | None = None, *, client: Any | None = None, pipeline: Any | None = None) -> None:
        self.config = config or LLMConfig()
        self._client = client
        self._pipeline = pipeline

    def stream(self, request: LabelRequest) -> Iterable[str]:
        yield self.label(request)

    def label(self, request: LabelRequest) -> str:
        texts = [candidate.text for candidate in request.candidates]
        prompt = _label_prompt(texts, max_words=request.max_words)
        if self.config.provider == "openai":
            client = self._client or self._openai_client()
            resp = client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=24,
            )
            text = resp.choices[0].message.content or ""
        elif self.config.provider == "local":
            pipe = self._pipeline or self._local_pipeline()
            out = pipe(prompt, max_new_tokens=24, do_sample=False, return_full_text=False)
            text = out[0]["generated_text"] if isinstance(out, list) else str(out)
        else:
            text = ""
        return _clean_label(text, request.max_words)

    def _openai_client(self) -> Any:
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover
            raise ImportError('LLMLabeler(provider="openai") needs the "openai" extra') from e
        kwargs: dict[str, Any] = {}
        if self.config.api_key:
            kwargs["api_key"] = self.config.api_key
        if self.config.base_url:
            kwargs["base_url"] = self.config.base_url
        self._client = OpenAI(**kwargs)
        return self._client

    def _local_pipeline(self) -> Any:
        try:
            from transformers import pipeline
        except ImportError as e:  # pragma: no cover
            raise ImportError('LLMLabeler(provider="local") needs the "local" extra') from e
        self._pipeline = pipeline("text-generation", model=self.config.model)
        return self._pipeline


def _label_prompt(texts: list[str], *, max_words: int) -> str:
    samples = "\n".join(f"- {text}" for text in texts[:20])
    return (
        f"Name this cluster in at most {max_words} words. "
        "Return only the label, no punctuation or explanation.\n"
        f"{samples}"
    )


def _clean_label(text: str, max_words: int) -> str:
    text = text.strip().splitlines()[0].strip(" .:-\"'")
    words = text.split()
    return " ".join(words[:max_words])
