# API Reference

This is the 0.1 public API. The package is organized around embedders,
labelers, loaders, persisters, representation models, and reconcilers.

## EmbedTree

```python
EmbedTree(
    embedder,
    config=None,
    *,
    state=None,
    labeler=None,
)
```

- `embedder`: callable `content -> vector`, or an object with `embed_batch`.
- `config`: optional `TreeConfig`.
- `state`: optional object with `load()` and `save(state)` for materialized tree state.
- `labeler`: optional `Labeler` used for node labels.

### Insert

```python
tree.add(content, item_id=None, payload=None, text=None)
tree.add_batch(contents, item_ids=None, payloads=None, texts=None)
tree.add_node(content_node)
tree.add_nodes(content_nodes)
tree.add_partial_tree(partial_tree)
```

`add_batch` uses `embedder.embed_batch(contents)` when available and persists
once after the batch. If `text` is omitted and content is a string, the content
is used as display text.

### Organize and Label

```python
tree.rebalance()
tree.label(labeler=None)
tree.organize(labeler=None)
```

- `rebalance()` rebuilds a clean top-down hierarchy from all current items.
- `label()` assigns human-readable labels to the current tree.
- `organize()` is `rebalance()` followed by `label()`.

### Query

```python
tree.query(content, k=10, exhaustive=False)
```

Returns `(item_id, distance, payload)` tuples sorted nearest first.

- `exhaustive=False`: route to one leaf, then rank candidates in that leaf.
- `exhaustive=True`: scan every item for exact nearest neighbors.

### Delete and Browse

```python
tree.remove(item_id)
tree.remove_batch(item_ids)
tree.show(max_items=3, collapse_single_leaf=False)
tree.to_dict(max_items=5, collapse_single_leaf=False)
tree.get_tree()
len(tree)
```

Deletion is local and does not re-embed, relabel, or recluster. Run
`rebalance()` after heavy churn if you want a fresh hierarchy.

## Configuration

```python
TreeConfig(
    max_branches=5,
    leaf_capacity=10,
    split_algo="kmeans",
    pca_dims=None,
    pca_mode="freeze",
    pca_warmup=1000,
    pca_batch_size=256,
    rebalance=RebalanceConfig(),
    llm=LLMConfig(),
    model_args={},
)
```

`TreeConfig` is a plain Pydantic model. It is explicit and does not load from
environment variables.

## Text Embedders

Bundled embedders are callable and can be passed directly as `embedder`.

```python
TagSetEmbedder(["docs", "ingest", "analysis"])

OpenAITextEmbedder(
    model="text-embedding-3-small",
    api_key="...",
    dimensions=None,
)

HuggingFaceTextEmbedder(
    model="BAAI/bge-small-en-v1.5",
    device="auto",
)
```

Shared embedder methods:

```python
embedder.embed(text)
embedder.embed_batch(texts)
embedder(text)
embedder.cache_clear()
```

To add an embedder, subclass `BaseTextEmbedder` and implement
`_embed_batch(texts) -> np.ndarray`, or pass any callable that returns a vector.

## Labelers

```python
KeywordLabeler()
LLMLabeler(config)
FunctionLabeler(fn)
```

`Labeler` implementations receive a `LabelRequest`:

```python
LabelRequest(
    candidates=[LabelCandidate(id="a", text="example")],
    max_words=6,
)
```

`FunctionLabeler` adapts a function that accepts a `LabelRequest` and returns a
string or chunks of a string.

## Persistence

For `EmbedTree(state=...)`, pass an object that can load and save materialized
state. `JsonTreeLoader` is the built-in JSON implementation:

```python
state = JsonTreeLoader("./tree.json")
tree = EmbedTree(embedder=embedder, state=state)
```

For source ingestion and export, use loaders and persisters with `PartialTree`.

## Representation

```python
ContentNode(id, content, text=None, payload=None, version=None)
KeyNode(id, label=None, payload=None, version=None)
TreeEdge(parent_id, child_id)
NodeEmbedding(node_id, vector, raw=None, model=None, version=None)
NodeAggregate(node_id, vsum, count, version=None)
PartialTree(...)
```

Loader contract:

```python
class TreeLoader(Protocol):
    def load(self) -> PartialTree | None: ...
```

Persister contract:

```python
class TreePersister(Protocol):
    def save(self, state) -> None: ...
```

Included implementations cover JSON files, filesystem folders, SQLite, and
SQLAlchemy-backed sources where the matching optional dependencies are installed.

## Reconciliation

```python
DefaultTreeReconciler().reconcile(
    ground_truth_loader,
    reusable_loader=None,
    embedder=embedder,
    config=config,
)
```

`TreeReconciler` builds reusable representation state from source data plus
optional cached embeddings, labels, or aggregates.

## PCA

PCA is disabled by default. Enable it for larger collections where lower
dimensional routing/clustering is useful.

```python
TreeConfig(
    pca_dims=64,
    pca_mode="freeze",
    pca_warmup=2000,
    pca_batch_size=512,
)
```

- `freeze`: fit PCA after warmup; `rebalance()` refits from all current items.
- `incremental`: update PCA in batches; `rebalance()` keeps the running projector.

During warmup, items are buffered and queries scan the buffer directly.
