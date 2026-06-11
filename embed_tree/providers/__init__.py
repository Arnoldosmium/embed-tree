"""Embedding providers: callable adapters usable as EmbedTree(embedder=...).

    from embed_tree.providers import OpenAIEmbeddingProvider
    embedder = OpenAIEmbeddingProvider(api_key="sk-...")

Concrete adapters import their heavy optional deps lazily, so importing this
package never requires openai or sentence-transformers.
"""

from .base import EmbeddingProvider
from .fake import FakeEmbeddingProvider
from .local import SentenceTransformerProvider
from .openai import OpenAIEmbeddingProvider

__all__ = [
    "EmbeddingProvider",
    "FakeEmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "SentenceTransformerProvider",
]
