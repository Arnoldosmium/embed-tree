"""Tiny runnable demo: cluster 2D points, persist, reload, query.

    python examples/demo.py

Distance is cosine (vectors are L2-normalized), so the three clusters are
distinguished by *direction* (angle), not position — hence centers that point
different ways rather than one sitting at the origin.
"""

import numpy as np

from embed_tree import ContentNode, EmbedTree, JsonTreeLoader, TreeConfig


def main():
    rng = np.random.default_rng(42)
    centers = {"A": [10, 1], "B": [10, 10], "C": [1, 10]}  # ~6deg / 45deg / ~84deg

    def embed(text):
        label, _ = text.split(":")
        return np.array(centers[label], dtype=float) + rng.normal(scale=0.4, size=2)

    tree = EmbedTree(
        embedder=embed,
        state=JsonTreeLoader("./demo_tree.json"),
        config=TreeConfig(leaf_capacity=15, max_branches=3),
    )

    for label in centers:
        for i in range(40):
            tree.add_node(ContentNode(f"{label}:{i}", f"{label}:{i}", {"label": label}))

    print(f"inserted {len(tree)} items")

    def near(label):
        v = np.array(centers[label], dtype=float)
        hits = tree.query(f"{label}:999", k=5)
        labels = [p["label"] for _, _, p in hits]
        print(f"query near {label} ({v.tolist()}): top-5 labels = {labels}")

    for label in centers:
        near(label)


if __name__ == "__main__":
    main()
