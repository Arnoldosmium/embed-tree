# embed-tree

`embed-tree` turns content embeddings into a browsable, labeled hierarchy.
It is useful when you have documents, notes, records, or search results and
want a compact taxonomy that a person can inspect.

The package is model-agnostic: you provide an embedder, and `embed-tree`
handles clustering, labeling, querying, deletion, and persistence.

## Install

```bash
pip install embed-tree
```

Optional integrations:

```bash
pip install "embed-tree[openai]"  # OpenAI embeddings and labels
pip install "embed-tree[local]"   # local Hugging Face embeddings/labels
pip install "embed-tree[sql]"     # SQLAlchemy loaders/persisters
```

## Quick Start

```python
from embed_tree import EmbedTree, TagSetEmbedder, TreeConfig

tree = EmbedTree(
    embedder=TagSetEmbedder(["docs", "ingest", "analysis", "schemas"]),
    config=TreeConfig(max_branches=5, leaf_capacity=10),
)

tree.add_batch(
    [
        {"tags": ["docs", "ingest"]},
        {"tags": ["ingest"]},
        {"tags": ["analysis"]},
        {"tags": ["docs", "schemas"]},
    ],
    texts=[
        "Write import pipeline documentation",
        "Add retry handling to data ingestion",
        "Reduce summary generation latency",
        "Document schema mapping examples",
    ],
)

tree.organize()
print(tree.show())
```

Use a real embedder in production:

```python
from embed_tree import EmbedTree, OpenAITextEmbedder

tree = EmbedTree(
    embedder=OpenAITextEmbedder(model="text-embedding-3-small", api_key="..."),
)
tree.add("Some document text", payload={"source": "docs"})
```

## Core API

```python
tree = EmbedTree(embedder, config=None, *, state=None, labeler=None)

tree.add(content, item_id=None, payload=None, text=None)
tree.add_batch(contents, item_ids=None, payloads=None, texts=None)
tree.add_node(content_node)
tree.add_nodes(content_nodes)
tree.add_partial_tree(partial_tree)

tree.organize(labeler=None)
tree.rebalance()
tree.label(labeler=None)

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

## Persistence

Use a loader that can also save materialized state:

```python
from embed_tree import EmbedTree, JsonTreeLoader

tree = EmbedTree(
    embedder=embedder,
    state=JsonTreeLoader("./tree.json"),
)
```

`JsonTreeLoader` writes an atomic JSON snapshot and reloads it when a new tree
is constructed with the same path.

## Labeling

By default, node labels are generated locally with `KeywordLabeler`. For LLM
labels, configure `TreeConfig.llm`:

```python
from embed_tree import LLMConfig, TreeConfig

config = TreeConfig(
    llm=LLMConfig(provider="openai", model="gpt-4o-mini", api_key="...")
)
```

For custom labels, pass a `Labeler` implementation. A small function can be
adapted with `FunctionLabeler`.

## More Documentation

See [docs/API.md](docs/API.md) for the fuller API reference, loader/persister
abstractions, PCA options, and extension points.

## Development

```bash
uv sync --extra dev
uv run --extra dev pytest -q
```
