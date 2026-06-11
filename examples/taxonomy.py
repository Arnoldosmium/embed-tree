"""Organize a handful of "open PRs" into a browsable, labeled taxonomy.

    python examples/taxonomy.py

Uses a local sentence-transformers model if available; otherwise a tiny
keyword-based fake embedder so the demo runs with no downloads. Labels come
from the no-network KeywordTagger (set config.llm to use an LLM instead).
"""

import numpy as np

from embed_tree import EmbedTree, TreeConfig

PRS = [
    "Fix null pointer in auth token refresh",
    "Add retry/backoff to auth login flow",
    "Auth: rotate session keys on password change",
    "Speed up dashboard query with index",
    "Cache dashboard aggregates to cut latency",
    "Reduce N+1 queries on dashboard load",
    "Bump numpy and scikit-learn versions",
    "Upgrade pytest and fix deprecation warnings",
    "Pin transitive deps in lockfile",
    "Document the embedding provider API",
    "Add README section on rebalancing",
    "Write docstrings for the tree module",
]


def make_embedder():
    """Real local embeddings if installed, else a keyword-bucket fake."""
    try:
        from embed_tree import SentenceTransformerProvider

        return SentenceTransformerProvider("BAAI/bge-small-en-v1.5")
    except Exception:
        buckets = {"auth": [1, 0, 0, 0], "dashboard": [0, 1, 0, 0],
                   "deps": [0, 0, 1, 0], "doc": [0, 0, 0, 1]}
        keys = {"auth": "auth", "dashboard": "dashboard", "deps": "dep",
                "doc": "document"}  # crude keyword match
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
    tree.add_batch(PRS)          # each PR's text defaults from the content string
    tree.organize()              # rebuild clean taxonomy + label nodes (KeywordTagger)

    print(tree.show(max_items=3))


if __name__ == "__main__":
    main()
