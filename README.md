# embed-tree

`embed-tree` turns content nodes into a browsable, labeled hierarchy.

The public model is intentionally small:

```python
ContentNode(id=..., text=..., metadata={...})
BranchNode(id, label=None, children=[])
EmbedTree(embedder, config=None, state=None, labeler=None)
```

`ContentNode.text` is the string passed to the embedder. `metadata` is opaque
user data returned by queries and preserved in exported branches.

## Install

```bash
pip install embed-tree
```

Optional integrations:

```bash
pip install "embed-tree[openai]"
pip install "embed-tree[local]"
pip install "embed-tree[sql]"
```

## Quick Start

```python
from embed_tree import ContentNode, EmbedTree, TagSetEmbedder, TreeConfig

nodes = [
    ContentNode(id="doc-1", text="import pipeline docs", metadata={"tags": ["docs", "ingest"]}),
    ContentNode(id="doc-2", text="retry handling for ingestion", metadata={"tags": ["ingest"]}),
    ContentNode(id="doc-3", text="summary generation latency", metadata={"tags": ["analysis"]}),
    ContentNode(id="doc-4", text="schema mapping examples", metadata={"tags": ["docs", "schemas"]}),
]

tree = EmbedTree(
    embedder=TagSetEmbedder(["docs", "ingest", "analysis", "schemas"]),
    config=TreeConfig(max_branches=4, leaf_capacity=2),
)

tree.add_nodes(nodes)
tree.organize()  # rebalance the hierarchy, then label each branch

print(tree.show())
branch = tree.to_branch()
```

Use a real text embedder in production:

```python
from embed_tree import ContentNode, EmbedTree, OpenAITextEmbedder

tree = EmbedTree(OpenAITextEmbedder(model="text-embedding-3-small", api_key="..."))
tree.add_node(
    ContentNode(
        id="doc-1",
        text="Some document summary",
        metadata={"source": "docs"},
    )
)
```

## Core API

```python
tree.add_node(ContentNode(...))      # -> id
tree.add_nodes([ContentNode(...)])   # -> list[id]
tree.add_branch(BranchNode(...))     # -> list[id], inserts all content leaves

tree.query("query text", k=10, exhaustive=False)
tree.remove(node_id)
tree.remove_batch([node_id])

tree.rebalance()
tree.label(labeler=None)
tree.organize(labeler=None) # rebalance + re-label

tree.to_branch(max_items=None)
tree.show(max_items=3)
len(tree)
```

`BranchNode` is the public tree shape. It can represent an input branch from a
loader or the organized output from `EmbedTree.to_branch()`.

For folder-based trees, `FileSystemTreeLoader` uses the file content MD5 as
`id`. Its optional `text_generator(path, raw_text)` can derive the embed text
from raw file text while preserving file identity. Its optional
`additional_metadata_derivers` is a list of callables that derive metadata such
as `new_file_name` from file content; derived dictionaries are merged in order,
with later keys winning. `FolderTreePersister` moves existing files only when a
node has a content MD5 as its `id` or explicit MD5 metadata and that MD5 exists
under the current root. If no current file matches, path metadata can point to a
source file to copy when its MD5 matches the same identity. If neither exists,
`missing_node_file` controls the result: `"skip"` warns and skips by default,
`"create"` writes a `.txt` snapshot containing `text` and `metadata`, and
`"raise"` raises `MissingNodeFileError`. `new_file_name` can rename moved/copied
files or snapshots.

`EmbedTree` has internal runtime nodes and content records which are not
public API.

## Persistence

Use a state loader that can save materialized state:

```python
from embed_tree import EmbedTree, JsonTreeLoader

tree = EmbedTree(embedder, state=JsonTreeLoader("./tree.json"))
```

## Development

```bash
uv sync --extra dev
uv run --extra dev pytest -q
```
