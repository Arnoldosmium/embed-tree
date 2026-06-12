# API Reference

## Models

```python
ContentNode(
    id,
    text: str,
    metadata: dict[str, Any] = {},
    embedding: Sequence[float] | None = None,
)
```

`text` is the string that gets embedded. `embedding` is an optional cache; when
present, `EmbedTree.add_node(s)` uses it instead of calling the embedder.

```python
BranchNode(
    id,
    label: str | None = None,
    children: list[BranchNode | ContentNode] = [],
    metadata: dict[str, Any] = {},
    vector_sum: Sequence[float] | None = None,
)
```

`BranchNode.count` is a property. If explicitly set, that cached value is used;
otherwise it is computed from children.

## EmbedTree

```python
EmbedTree(embedder, config=None, *, state=None, labeler=None)
```

### Insert

```python
tree.add_node(node)
tree.add_nodes(nodes)
tree.add_branch(branch)
```

### Organize and Label

```python
tree.rebalance()
tree.label(labeler=None)
tree.organize(labeler=None)
```

### Query and Delete

```python
tree.query(text, k=10, exhaustive=False)
tree.remove(node_id)
tree.remove_batch(node_ids)
```

Query returns `(id, distance, metadata)` tuples.

### Export

```python
tree.to_branch(max_items=None, collapse_single_leaf=False)
tree.show(max_items=3, collapse_single_leaf=False)
len(tree)
```

`to_branch()` returns the public recursive tree shape.

## Embedders

```python
TagSetEmbedder(["docs", "ingest", "analysis"])
OpenAITextEmbedder(model="text-embedding-3-small", api_key="...")
HuggingFaceTextEmbedder(model="BAAI/bge-small-en-v1.5")
```

Custom embedders can subclass `BaseTextEmbedder` or be any callable accepting
text and returning a vector.
