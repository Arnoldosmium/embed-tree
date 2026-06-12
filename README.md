# embed-tree

`embed-tree` turns content nodes into a browsable, labeled hierarchy.

The public model is intentionally small:

```python
ContentNode(id, text, metadata={})
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
    ContentNode("doc-1", "import pipeline docs", {"tags": ["docs", "ingest"]}),
    ContentNode("doc-2", "retry handling for ingestion", {"tags": ["ingest"]}),
    ContentNode("doc-3", "summary generation latency", {"tags": ["analysis"]}),
    ContentNode("doc-4", "schema mapping examples", {"tags": ["docs", "schemas"]}),
]

tree = EmbedTree(
    embedder=TagSetEmbedder(["docs", "ingest", "analysis", "schemas"]),
    config=TreeConfig(max_branches=4, leaf_capacity=2),
)

tree.add_nodes(nodes)
tree.organize()

print(tree.show())
branch = tree.to_branch()
```

Use a real text embedder in production:

```python
from embed_tree import ContentNode, EmbedTree, OpenAITextEmbedder

tree = EmbedTree(OpenAITextEmbedder(model="text-embedding-3-small", api_key="..."))
tree.add_node(ContentNode("doc-1", "Some document summary", {"source": "docs"}))
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
tree.organize(labeler=None)

tree.to_branch(max_items=None)
tree.show(max_items=3)
len(tree)
```

`BranchNode` is the public tree shape. It can represent an input branch from a
loader or the organized output from `EmbedTree.to_branch()`.

`EmbedTree` has internal runtime nodes and content records, but they are not
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
