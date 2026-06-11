"""Node taggers: turn a cluster's member texts into a short, readable label.

See DESIGN.md §10. A tagger is any `Callable[[list[str]], str]`. Three ways:

  - KeywordTagger  : no network/deps beyond sklearn; TF-IDF top terms.
  - LLMTagger      : generate a label with an LLM (OpenAI, or a local model /
                     OpenAI-compatible server), driven by `LLMConfig`.
  - your own       : pass any callable as `EmbedTree(tagger=...)`.

`make_tagger(llm_config)` builds the right one from config (provider "none"
=> KeywordTagger).
"""

from __future__ import annotations

from typing import Any, Callable

from .config import LLMConfig

Tagger = Callable[[list[str]], str]


def make_tagger(config: LLMConfig, *, client: Any | None = None, pipeline: Any | None = None) -> Tagger:
    if config.provider == "none":
        return KeywordTagger()
    return LLMTagger(config, client=client, pipeline=pipeline)


class KeywordTagger:
    """Label a cluster by its most distinctive terms (TF-IDF). No network."""

    def __init__(self, top_k: int = 3) -> None:
        self.top_k = top_k

    def __call__(self, texts: list[str]) -> str:
        texts = [t for t in texts if t and t.strip()]
        if not texts:
            return ""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            vec = TfidfVectorizer(stop_words="english", max_features=2000)
            X = vec.fit_transform(texts)
            import numpy as np

            scores = np.asarray(X.sum(axis=0)).ravel()
            terms = np.asarray(vec.get_feature_names_out())
            top = terms[scores.argsort()[::-1][: self.top_k]]
            label = ", ".join(top)
            return label or _truncate(texts[0])
        except ValueError:
            # empty vocabulary (e.g. all stop words / no alpha tokens)
            return _truncate(texts[0])


class LLMTagger:
    """Generate a concise cluster label with an LLM.

    provider="openai": uses the OpenAI client (set `base_url` for an
    OpenAI-compatible local server, e.g. Ollama/vLLM/LM Studio).
    provider="local":  uses a transformers text-generation pipeline (HF).
    Inject `client`/`pipeline` to avoid network in tests.
    """

    def __init__(self, config: LLMConfig, *, client: Any | None = None, pipeline: Any | None = None) -> None:
        self.config = config
        self._client = client
        self._pipeline = pipeline

    def __call__(self, texts: list[str]) -> str:
        texts = [t for t in texts if t and t.strip()][: self.config.max_samples]
        if not texts:
            return ""
        prompt = self._prompt(texts)
        raw = self._generate(prompt)
        return _clean_label(raw, self.config.max_label_words)

    def _prompt(self, texts: list[str]) -> str:
        bullets = "\n".join(f"- {t}" for t in texts)
        return (
            f"These items belong to one group. Give a concise topic label of at "
            f"most {self.config.max_label_words} words that describes what they have "
            f"in common. Reply with only the label.\n\n{bullets}\n\nLabel:"
        )

    def _generate(self, prompt: str) -> str:
        if self.config.provider == "openai":
            client = self._client or self._openai_client()
            resp = client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=24,
            )
            return resp.choices[0].message.content or ""
        # local transformers pipeline
        pipe = self._pipeline or self._local_pipeline()
        out = pipe(prompt, max_new_tokens=24, do_sample=False, return_full_text=False)
        return out[0]["generated_text"]

    def _openai_client(self) -> Any:
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover
            raise ImportError('LLMTagger(provider="openai") needs the "openai" extra') from e
        kwargs: dict[str, Any] = {}
        if self.config.api_key:
            kwargs["api_key"] = self.config.api_key
        if self.config.base_url:
            kwargs["base_url"] = self.config.base_url
        self._client = OpenAI(**kwargs)
        return self._client

    def _local_pipeline(self) -> Any:
        try:
            from transformers import pipeline as hf_pipeline
        except ImportError as e:  # pragma: no cover
            raise ImportError('LLMTagger(provider="local") needs the "local" extra') from e
        self._pipeline = hf_pipeline("text-generation", model=self.config.model)
        return self._pipeline


def _truncate(text: str, n: int = 40) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _clean_label(text: str, max_words: int) -> str:
    line = text.strip().splitlines()[0] if text.strip() else ""
    line = line.strip().strip('"').strip("'").rstrip(".")
    words = line.split()
    return " ".join(words[:max_words])
