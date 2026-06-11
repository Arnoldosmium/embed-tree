# embed-tree

`embed-tree` turns content embeddings into a browsable, labeled hierarchy.
It is useful when you have documents, notes, tickets, search results, or any
other text-like records and want a small taxonomy that a person can inspect.

The library stays model-agnostic: you provide an embedder, and `embed-tree`
handles clustering, labeling, querying, deletion, and persistence.

## Install

```bash
pip install embed-tree
```

Optional adapters:

```bash
pip install "embed-tree[openai]"  # OpenAI embeddings
pip install "embed-tree[local]"   # sentence-transformers embeddings
pip install "embed-tree[sql]"     # SQLAlchemy loaders/persisters
```

## Quick Start

```python
from embed_tree import EmbedTree, FakeEmbeddingProvider, TreeConfig

embedder = FakeEmbeddingProvider(dim=32)  # deterministic demo embedder
tree = EmbedTree(
    embedder=embedder,
    config=TreeConfig(max_branches=5, leaf_capacity=10),
)

tree.add_batch(
    [
        "Write import pipeline documentation",
        "Fix login session refresh",
        "Reduce report query latency",
        "Add retry handling to data ingestion",
    ]
)

tree.organize()       # rebuild a clean hierarchy and label every node
print(tree.show())    # human-readable outline
```

Use a real embedding provider in production:

```python
from embed_tree import EmbedTree, OpenAIEmbeddingProvider

embedder = OpenAIEmbeddingProvider(
    model="text-embedding-3-small",
    api_key="...",
)

tree = EmbedTree(embedder=embedder)
tree.add("Some document text", payload={"source": "docs"})
```

## Core API

`EmbedTree` is the main entry point.

```python
tree.add(content, item_id=None, payload=None, text=None)
tree.add_batch(contents, item_ids=None, payloads=None, texts=None)
tree.add_node(content_node)
tree.add_nodes(content_nodes)
tree.add_partial_tree(partial_tree)

tree.organize(tagger=None)
tree.rebalance()
tree.label(tagger=None)

tree.query(content, k=10, exhaustive=False)
tree.remove(item_id)
tree.remove_batch(item_ids)

tree.show(max_items=3)
tree.to_dict(max_items=5)
tree.get_tree()
len(tree)
```

`content` is what gets embedded. `text` is the human-readable string used in
labels and browse output; it defaults to `content` when `content` is a string.
`payload` is returned in query results and exported browse data.

## Configuration

Configuration is explicit and code-driven through `TreeConfig`. It does not
read environment variables.

```python
from embed_tree import LLMConfig, RebalanceConfig, TreeConfig

config = TreeConfig(
    max_branches=5,
    leaf_capacity=10,
    rebalance=RebalanceConfig(enabled=True, every_n_inserts=10_000),
    llm=LLMConfig(provider="none"),  # default keyword labels, no network
)
```

Defaults are tuned for readable taxonomies: small fan-out and small leaves.
Raise `max_branches` and `leaf_capacity` when using the tree primarily as a
retrieval index.

## Querying

```python
hits = tree.query("related content", k=5)
# [(item_id, distance, payload), ...]

exact_hits = tree.query("related content", k=5, exhaustive=True)
```

Default queries route to one leaf and rank items there, which is fast but
approximate. `exhaustive=True` scans every item for exact nearest neighbors.

## Persistence

```python
from embed_tree import EmbedTree, FileTreeStore

tree = EmbedTree(
    embedder=embedder,
    store=FileTreeStore("./tree.json"),
)
```

`FileTreeStore` saves an atomic JSON snapshot after writes and reloads it when
the tree is constructed again.

## Labeling

Without extra configuration, node labels are generated locally from keywords.
For LLM labels:

```python
from embed_tree import LLMConfig, TreeConfig

config = TreeConfig(
    llm=LLMConfig(provider="openai", model="gpt-4o-mini", api_key="...")
)
```

You can also pass a custom `tagger: Callable[[list[str]], str]` to
`EmbedTree(..., tagger=...)`, `tree.label(tagger=...)`, or
`tree.organize(tagger=...)`.

## More Documentation

See [docs/API.md](docs/API.md) for the fuller API reference, provider details,
loader/persister abstractions, PCA options, and extension points.

## Development

```bash
uv sync --extra dev
uv run pytest -q
```
