"""Organize a handful of notes into a browsable, labeled taxonomy.

    python examples/taxonomy.py

Uses a local sentence-transformers model if available; otherwise a tiny
keyword-based fake embedder so the demo runs with no downloads. Labels come
from the no-network KeywordLabeler (set config.llm to use an LLM instead).
"""

import numpy as np

from embed_tree import EmbedTree, TreeConfig

DOCS = [
    "Write import pipeline documentation",
    "Add retry handling to data ingestion",
    "Document schema mapping examples",
    "Reduce summary generation latency",
    "Cache repeated analysis results",
    "Batch expensive text processing calls",
    "Bump numpy and scikit-learn versions",
    "Upgrade pytest and fix deprecation warnings",
    "Pin transitive dependencies in lockfile",
    "Document the text embedder API",
    "Add README section on rebalancing",
    "Write docstrings for the tree module",
]


def make_embedder():
    """Real local embeddings if installed, else a keyword-bucket fake."""
    try:
        from embed_tree import HuggingFaceTextEmbedder

        return HuggingFaceTextEmbedder("BAAI/bge-small-en-v1.5")
    except Exception:
        buckets = {"ingest": [1, 0, 0, 0], "analysis": [0, 1, 0, 0],
                   "deps": [0, 0, 1, 0], "docs": [0, 0, 0, 1]}
        keys = {"ingest": "ingest", "analysis": "analysis", "deps": "dep",
                "docs": "doc"}  # crude keyword match
        def embed(text):
            t = text.lower()
            for name, base in buckets.items():
                if keys[name] in t or name in t:
                    return np.array(base, float) + 0.01 * (abs(hash(text)) % 100) / 100
            return np.zeros(4)
        return embed


def main():
    tree = EmbedTree(
        embedder=make_embedder(),
        config=TreeConfig(max_branches=4, leaf_capacity=4, model_args={"random_state": 0}),
    )
    tree.add_batch(DOCS)         # each item's text defaults from the content string
    tree.organize()              # rebuild clean taxonomy + label nodes

    print(tree.show(max_items=3))


if __name__ == "__main__":
    main()
