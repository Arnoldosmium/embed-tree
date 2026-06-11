# API Reference

This reference covers the public API exported from `embed_tree`.

## EmbedTree

```python
EmbedTree(
    embedder,
    store=None,
    config=None,
    *,
    tagger=None,
)
```

- `embedder`: callable `content -> vector`, or an object with `embed_batch`.
- `store`: optional `TreeStore`; defaults to in-memory `NullTreeStore`.
- `config`: optional `TreeConfig`.
- `tagger`: optional callable `list[str] -> str` used for node labels.

### Insert

```python
item_id = tree.add(content, item_id=None, payload=None, text=None)
item_ids = tree.add_batch(contents, item_ids=None, payloads=None, texts=None)
```

`add_batch` uses `embedder.embed_batch(contents)` when available and persists
once after the batch. If `text` is omitted and content is a string, the content
is used as display text.

Loader-compatible helpers:

```python
tree.add_node(content_node)
tree.add_nodes(content_nodes)
tree.add_partial_tree(partial_tree)
```

### Organize and Label

```python
tree.rebalance()
tree.label(tagger=None)
tree.organize(tagger=None)
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

### Delete

```python
tree.remove(item_id)          # -> bool
tree.remove_batch(item_ids)   # -> count removed
```

Deletion is local and does not re-embed, relabel, or recluster. Run
`rebalance()` after heavy churn if you want a fresh hierarchy.

### Browse and Inspect

```python
tree.show(max_items=3, collapse_single_leaf=False)
tree.to_dict(max_items=5, collapse_single_leaf=False)
tree.get_tree()
len(tree)
```

`show()` returns an indented outline. `to_dict()` returns nested dictionaries
with labels, sizes, children, and item previews for UI rendering.

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

### RebalanceConfig

```python
RebalanceConfig(
    enabled=True,
    every_n_inserts=10_000,
    on_demand=True,
)
```

When `every_n_inserts` is set, the tree rebuilds after that many inserts.
Manual `tree.rebalance()` is always the direct way to rebuild.

### LLMConfig

```python
LLMConfig(
    provider="none",       # "none" | "openai" | "local"
    model="gpt-4o-mini",
    api_key=None,
    base_url=None,
    max_samples=15,
    max_label_words=6,
)
```

- `provider="none"` uses local keyword labels.
- `provider="openai"` uses the OpenAI SDK or an OpenAI-compatible endpoint.
- `provider="local"` uses a local transformers pipeline.

## Embedding Providers

Bundled providers are callable and can be passed directly as `embedder`.

```python
OpenAIEmbeddingProvider(
    model="text-embedding-3-small",
    api_key="...",
    dimensions=None,
    cache=True,
    normalize=False,
    max_retries=5,
    backoff_base=0.5,
)

SentenceTransformerProvider(
    model_name="BAAI/bge-small-en-v1.5",
    cache=True,
)

FakeEmbeddingProvider(dim=32)
```

Provider methods:

```python
provider.embed(text)
provider.embed_batch(texts)
provider(text)
provider.cache_clear()
```

To add a provider, subclass `EmbeddingProvider` and implement
`_embed_batch(texts) -> np.ndarray`; the base class supplies caching, batching,
retry, and optional normalization.

## Persistence

`TreeStore` is the compatibility snapshot API used by `EmbedTree(store=...)`.

```python
class TreeStore(Protocol):
    def load(self) -> TreeState | None: ...
    def save(self, state: TreeState) -> None: ...
```

Included stores:

```python
FileTreeStore(path)
NullTreeStore()
```

`FileTreeStore` writes a complete JSON snapshot atomically.

## Loader and Persister Abstractions

Newer ingestion/export flows use storage-neutral models.

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

`TreeReconciler` builds operational state from source data plus optional
reusable state.

```python
DefaultTreeReconciler().reconcile(
    ground_truth_loader,
    reusable_loader=None,
    embedder=embedder,
    config=config,
)
```

This is useful when content comes from one source while embeddings, labels, or
aggregates may be reused from another source.

## PCA

PCA is disabled by default. Enable it for larger collections where lower
dimensional routing/clustering is useful.

```python
TreeConfig(
    pca_dims=64,
    pca_mode="freeze",      # "freeze" | "incremental"
    pca_warmup=2000,
    pca_batch_size=512,
)
```

- `freeze`: fit PCA after warmup; `rebalance()` refits from all current items.
- `incremental`: update PCA in batches; `rebalance()` keeps the running projector.

During warmup, items are buffered and queries scan the buffer directly.

## Legacy and Compatibility Exports

The package also exports lower-level contracts and implementations used by the
current internals and compatibility layer:

- reducers: `Reducer`, `IdentityReducer`, `FreezePCAReducer`,
  `IncrementalPCAReducer`
- projectors: `PCAConfig`, `PCAProjector`, `VectorProjector`
- labelers: `LabelCandidate`, `LabelRequest`, `Labeler`, `FunctionLabeler`,
  `LLMLabeler`
- taggers: `Tagger`, `KeywordTagger`, `LLMTagger`, `make_tagger`
- representation helpers: `partial_tree_from_dict`, `partial_tree_to_dict`
