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

## Folder Trees

`FileSystemTreeLoader` uses file content MD5 as each file `ContentNode.id`.
It also stores `path`, `relative_path`, `filename`, and `version` in metadata.
Pass `text_generator=callable` to derive `ContentNode.text` from raw file text:

```python
FileSystemTreeLoader(
    "./docs",
    include_suffixes=[".md"],
    text_generator=lambda path, raw: raw.splitlines()[0],
    additional_metadata_derivers=[
        lambda raw: {"title": raw.splitlines()[0]},
        lambda raw: {"new_file_name": "derived.md"},
    ],
)
```

`additional_metadata_derivers` is a list of callables. Each callable receives
raw file text and returns a metadata mapping. Mappings are merged in order with
`|`, so later derivers override earlier keys. It is useful for fields such as
`new_file_name`, tags, titles, or source-specific identifiers derived from file
content.

`FolderTreePersister` builds a current-folder MD5 map and moves existing files
only when a node has a content MD5 as its `id` or explicit MD5 metadata such as
`md5`, `file_md5`, `content_md5`, or `content_id`. If no current file matches,
`path`, `relative_path`, or `source_path` metadata can point to a source file to
copy when its MD5 matches the same identity.

When neither move nor copy is possible, `missing_node_file` controls behavior:

```python
FolderTreePersister("./docs", missing_node_file="skip")   # default: warn + skip
FolderTreePersister("./docs", missing_node_file="create") # write .txt snapshot
FolderTreePersister("./docs", missing_node_file="raise")  # raise MissingNodeFileError
```

Created snapshots contain `text` and `metadata`. Set
`metadata["new_file_name"]` to rename a moved/copied file or snapshot.

## Embedders

```python
TagSetEmbedder(["docs", "ingest", "analysis"])
OpenAITextEmbedder(model="text-embedding-3-small", api_key="...")
HuggingFaceTextEmbedder(model="BAAI/bge-small-en-v1.5")
```

Custom embedders can subclass `BaseTextEmbedder` or be any callable accepting
text and returning a vector.
