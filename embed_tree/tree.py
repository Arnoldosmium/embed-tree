"""Core incremental hierarchical clustering tree.

Model:
  - Each content -> exactly one vector -> a unique leaf (strict tree, single
    membership).
  - A node caches an incremental centroid (vsum / count) for fast routing.
  - Incremental `add` routes a vector down to a leaf, then asks the configured
    split strategy whether/how to split once the leaf exceeds leaf_capacity.
  - `rebalance()` rebuilds a clean taxonomy top-down (divisive clustering),
    and `label()` names every node.

Vectors:
  - Each item keeps its *raw* embedding (source of truth, kept only when PCA is
    active) and the *routing* vector used in the tree.
  - With pca_dims set, the routing vector is the PCA-reduced raw. Until the PCA
    is fitted (warmup), items wait in a flat buffer and queries scan it directly.

Distance: routing vectors are always L2-normalized, so plain Euclidean ranks
identically to cosine everywhere (incl. KMeans). Magnitude is intentionally
ignored — embeddings encode meaning in direction.

PCA rebalance contract:
  - freeze      -> rebalance() refits PCA on all current raw vectors, then
                   re-projects everything and rebuilds the tree.
  - incremental -> rebalance() keeps the running PCA (flushing any pending
                   partial_fit), re-projects all raw vectors and rebuilds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Hashable, Iterable

import numpy as np

from .config import TreeConfig
from .reducers import Reducer
from .representation import BranchNode, ContentNode
from .labelers import KeywordLabeler, LabelCandidate, LabelRequest, Labeler, LLMLabeler
from .persisters import MaterializedTreeState
from .splitters import make_splitter

Content = Any
Vector = np.ndarray


@dataclass
class _StoredContent:
    id: Hashable
    vector: Vector  # routing vector (PCA-reduced if PCA active, else raw)
    metadata: Any = None
    raw: Vector | None = None  # original embedding; None means raw == vector
    text: str | None = None  # human-readable text, used for node labels / display


def _raw_of(item: _StoredContent) -> Vector:
    return item.raw if item.raw is not None else item.vector


@dataclass
class _TreeNode:
    id: int
    vsum: Vector  # running sum of descendant routing vectors
    count: int = 0
    children: list["_TreeNode"] | None = None
    items: list[_StoredContent] | None = field(default=None)
    unsplittable: bool = False  # leaf KMeans could not divide (e.g. dup vectors)
    label: str | None = None  # human-readable topic name (set by label())

    @property
    def is_leaf(self) -> bool:
        return self.children is None

    @property
    def centroid(self) -> Vector:
        return self.vsum / max(self.count, 1)


class EmbedTree:
    """Out-of-the-box incremental embedding tree.

    Inject an `embedder` (content -> vector) and optionally `state`
    persistence. Everything else is driven by `config`.
    """

    def __init__(
        self,
        embedder: Callable[[Content], Vector],
        config: TreeConfig | None = None,
        *,
        state: Any | None = None,
        labeler: Labeler | None = None,
    ) -> None:
        self.embedder = embedder
        self.state = state
        self.config = config or TreeConfig()
        self._labeler = labeler
        self._splitter = make_splitter(self.config)

        self.reducer: Reducer = Reducer.from_config(self.config)
        self._next_node_id = 0
        self._next_item_id = 0
        self._inserts_since_rebalance = 0
        self._warmup_buf: list[_StoredContent] = []  # items awaiting first PCA fit
        self._pf_buf: list[Vector] = []  # raw vectors pending partial_fit (incremental)
        self._reset_tree()

        snapshot = self._load_state()
        if snapshot is not None:
            self._restore(snapshot)

    # ---------------------------------------------------------------- public

    def add_node(self, node: ContentNode) -> Hashable:
        """Embed and insert one content leaf."""
        return self.add_nodes([node])[0]

    def add_nodes(self, nodes: Iterable[ContentNode]) -> list[Hashable]:
        """Embed and insert many content leaves, then persist once."""
        nodes = list(nodes)
        if not nodes:
            return []
        raws = self._vectors_for_nodes(nodes)
        ids: list[Hashable] = []
        for node, raw in zip(nodes, raws):
            r = self._prep_raw(raw)
            self._ingest(_StoredContent(node.id, vector=r, metadata=node.metadata, raw=r, text=node.text))
            ids.append(node.id)
        self._after_write(len(nodes))
        return ids

    def add_branch(self, branch: BranchNode) -> list[Hashable]:
        """Insert all content leaves from a branch."""
        return self.add_nodes(_content_leaves(branch))

    def _vectors_for_nodes(self, nodes: list[ContentNode]) -> list[Vector]:
        out: list[Vector | None] = [None] * len(nodes)
        missing_indexes: list[int] = []
        missing_texts: list[str] = []
        for i, node in enumerate(nodes):
            if node.embedding is None:
                missing_indexes.append(i)
                missing_texts.append(node.text)
            else:
                out[i] = np.asarray(node.embedding)
        if missing_texts:
            for i, vector in zip(missing_indexes, self._embed_many(missing_texts)):
                out[i] = vector
        return [np.asarray(vector) for vector in out if vector is not None]

    def _embed_many(self, contents: list[str]) -> list[Vector]:
        """Embed via the embedder's batch API if it has one, else one by one."""
        batch_fn = getattr(self.embedder, "embed_batch", None)
        if callable(batch_fn):
            return [np.asarray(v) for v in batch_fn(contents)]
        return [self.embedder(c) for c in contents]

    def _after_write(self, n: int) -> None:
        """Shared post-insert bookkeeping: rebalance-if-due, else persist once."""
        self._inserts_since_rebalance += n
        rb = self.config.rebalance
        if rb.enabled and rb.every_n_inserts and self._inserts_since_rebalance >= rb.every_n_inserts:
            self.rebalance()  # rebalance() persists
        else:
            self._save_state()

    def query(self, content: Content, k: int = 10, *, exhaustive: bool = False) -> list[tuple[Hashable, float, Any]]:
        """Return up to k nearest items as (item_id, distance, metadata).

        During PCA warmup (before the projection is fitted), all items live in a
        buffer and are scanned in raw space. Otherwise routes to a single leaf
        (approximate); exhaustive=True scans every item (exact).
        """
        raw = self._prep_raw(self.embedder(content))
        if self._in_warmup():
            return self._rank(raw, self._warmup_buf, k)
        q = self._route_vector(raw)
        items = self._all_items() if exhaustive else (self._route(q)[-1].items or [])
        return self._rank(q, items, k)

    def remove(self, item_id: Hashable) -> bool:
        """Delete an item by id. Returns True if it was found.

        Local, O(depth) operation — no re-embedding and no re-clustering. The
        cached routing vector is reused: we DFS to the leaf that holds the item,
        drop it, then walk the ancestor chain subtracting its vector from each
        `vsum` and decrementing `count` (so every `centroid` updates for free).
        Now-empty leaves are pruned so they can't swallow future queries.
        """
        removed = self._remove(item_id)
        if removed:
            self._save_state()
        return removed

    def remove_batch(self, item_ids: list[Hashable]) -> int:
        """Delete many items by id, persisting once. Returns the count removed."""
        n = sum(1 for iid in item_ids if self._remove(iid))
        if n:
            self._save_state()
        return n

    def _remove(self, item_id: Hashable) -> bool:
        """Remove an item without persisting. See remove()."""
        for i, it in enumerate(self._warmup_buf):  # still buffered (pre-PCA-fit)
            if it.id == item_id:
                self._warmup_buf.pop(i)
                return True

        path = self._find_path(self.root, item_id)
        if path is None:
            return False

        leaf = path[-1]
        item = next(it for it in (leaf.items or []) if it.id == item_id)
        leaf.items.remove(item)  # type: ignore[union-attr]
        for node in path:  # roll the item back out of every ancestor centroid
            node.vsum = node.vsum - item.vector
            node.count -= 1
        self._prune_empty(path)
        return True

    def _find_path(self, node: _TreeNode, item_id: Hashable) -> list[_TreeNode] | None:
        """DFS for the leaf holding item_id; return the root->leaf path or None."""
        if node.is_leaf:
            return [node] if any(it.id == item_id for it in (node.items or [])) else None
        for child in node.children or []:
            sub = self._find_path(child, item_id)
            if sub is not None:
                return [node, *sub]
        return None

    def _prune_empty(self, path: list[_TreeNode]) -> None:
        """Drop now-empty leaves bottom-up; a parent emptied of all children
        collapses back into an empty leaf. The root is always kept."""
        for parent, child in zip(reversed(path[:-1]), reversed(path[1:])):
            if not child.is_leaf or child.count > 0:
                break
            parent.children.remove(child)  # type: ignore[union-attr]
            if parent.children:
                break
            parent.children = None  # parent is now an empty leaf
            parent.items = []

    def rebalance(self) -> None:
        """Rebuild a clean taxonomy from all items via top-down divisive
        clustering, refitting PCA per the mode contract first.

        Unlike incremental `add`, this rebuilds the hierarchy top-down using
        the configured split strategy.
        """
        items = self._all_items() + self._warmup_buf
        if self.reducer.kind != "identity" and items:
            raws = np.stack([_raw_of(it) for it in items])
            if self.config.pca_mode == "freeze" or not self.reducer.is_fitted:
                self.reducer.fit(raws)  # freeze: full refit
            elif self._pf_buf:
                self.reducer.partial_fit(np.stack(self._pf_buf))  # flush pending
        self._pf_buf = []
        self._warmup_buf = []

        for it in items:  # re-project under the (possibly updated) reducer
            it.vector = self._route_vector(_raw_of(it))
            it.raw = None if self.reducer.kind == "identity" else _raw_of(it)

        self._next_node_id = 0
        self.root = self._divisive(items) if items else self._empty_root()
        self._inserts_since_rebalance = 0
        self._save_state()

    def label(self, labeler: Labeler | None = None) -> None:
        """Assign a human-readable label to every node from its members.

        Uses the injected `labeler`, else the one built from `config.llm`
        (provider "none" => TF-IDF keywords). Decoupled from rebuild because an
        LLM labeler is the expensive part. Call after rebalance() (or organize()).
        Labels are intentionally lazy: non-branching wrappers and their sole
        leaf child are left unlabeled because the browse output can inherit a
        readable title from the only child/item instead of paying for a gist.
        """
        labeler = labeler or self._labeler or self._make_labeler()
        self._label_node(self.root, labeler, sibling_count=0)
        self._save_state()

    def organize(self, labeler: Labeler | None = None) -> None:
        """One call: rebuild the clean taxonomy and label every node."""
        self.rebalance()
        self.label(labeler)

    def to_branch(self, *, max_items: int | None = None, collapse_single_leaf: bool = False) -> BranchNode:
        """Export the organized tree as public BranchNode/ContentNode data."""
        return self._to_branch(self.root, max_items, collapse_single_leaf)

    def show(self, *, max_items: int = 3, collapse_single_leaf: bool = False) -> str:
        """Pretty-print the taxonomy as an indented outline."""
        lines: list[str] = []
        self._show(self.root, 0, max_items, collapse_single_leaf, lines)
        return "\n".join(lines)

    def __len__(self) -> int:
        return self.root.count + len(self._warmup_buf)

    # --------------------------------------------------------- taxonomy build

    def _divisive(self, items: list[_StoredContent]) -> _TreeNode:
        """Recursively split items using the configured split strategy."""
        node = self._new_node()
        vecs = np.stack([it.vector for it in items])
        node.vsum = vecs.sum(axis=0)
        node.count = len(items)

        decision = self._splitter.split(items)
        if not decision.should_split:
            node.items = items
            node.unsplittable = decision.reason == "unsplittable"
            return node

        node.children = [self._divisive(cluster) for cluster in decision.clusters]
        return node

    def _empty_root(self) -> _TreeNode:
        root = self._new_node()
        root.items = []
        return root

    # ------------------------------------------------------------- labeling

    def _label_node(self, node: _TreeNode, labeler: Labeler, *, sibling_count: int) -> None:
        child_count = len(node.children or [])
        should_label = child_count > 1 or sibling_count > 1
        if should_label:
            candidates = self._label_candidates(node)
            request = LabelRequest(candidates=candidates, max_words=self.config.llm.max_label_words)
            node.label = labeler.label(request) if candidates else None
        else:
            node.label = None
        if not node.is_leaf:
            for child in node.children or []:
                self._label_node(child, labeler, sibling_count=child_count)

    def _label_candidates(self, node: _TreeNode) -> list[LabelCandidate]:
        items = [it for it in self._node_items(node) if it.text]
        if not items:
            return []
        c = node.centroid
        items.sort(key=lambda it: _dist(c, it.vector))  # closest to centroid first
        return [
            LabelCandidate(id=it.id, text=it.text or "", distance=_dist(c, it.vector), metadata=it.metadata)
            for it in items[: self.config.llm.max_samples]
        ]

    def _make_labeler(self) -> Labeler:
        if self.config.llm.provider == "none":
            return KeywordLabeler()
        return LLMLabeler(self.config.llm)

    # -------------------------------------------------------------- browsing

    def _to_branch(self, node: _TreeNode, max_items: int | None, collapse_single_leaf: bool) -> BranchNode:
        display_node = self._display_node(node, collapse_single_leaf)
        if display_node.is_leaf:
            items = display_node.items or []
            if max_items is not None:
                items = items[:max_items]
            children: list[BranchNode | ContentNode] = [
                ContentNode(id=it.id, text=it.text or str(it.id), metadata=dict(it.metadata or {}), embedding=it.vector.tolist())
                for it in items
            ]
        else:
            children = [self._to_branch(c, max_items, collapse_single_leaf) for c in node.children or []]
        branch = BranchNode(
            id=node.id,
            label=self._display_label(node),
            children=children,
            vector_sum=node.vsum.tolist(),
        )
        branch.count = node.count
        return branch

    def _show(
        self,
        node: _TreeNode,
        depth: int,
        max_items: int,
        collapse_single_leaf: bool,
        lines: list[str],
    ) -> None:
        indent = "  " * depth
        display_node = self._display_node(node, collapse_single_leaf)
        lines.append(f"{indent}- {self._display_label(node)} [{node.count}]")
        if display_node.is_leaf:
            for it in (display_node.items or [])[:max_items]:
                lines.append(f"{indent}    · {_short(it.text or str(it.id))}")
        else:
            for child in node.children or []:
                self._show(child, depth + 1, max_items, collapse_single_leaf, lines)

    def _display_node(self, node: _TreeNode, collapse_single_leaf: bool) -> _TreeNode:
        if not collapse_single_leaf:
            return node
        current = node
        while not current.is_leaf and len(current.children or []) == 1:
            child = current.children[0]
            if child.is_leaf:
                return child
            current = child
        return node

    def _display_label(self, node: _TreeNode) -> str:
        if node.label:
            return node.label
        if node.is_leaf:
            item = next((it for it in node.items or [] if it.text), None)
            return _short(item.text) if item is not None else "(unlabeled)"
        child = (node.children or [None])[0]
        return self._display_label(child) if child is not None else "(unlabeled)"

    # ---------------------------------------------------------- ingestion

    def _ingest(self, item: _StoredContent) -> None:
        # No PCA: route in raw space, raw == vector (store once).
        if self.reducer.kind == "identity":
            item.vector = self._prep_routing(item.vector)
            item.raw = None
            self._insert(item)
            return

        # PCA, not yet fitted: buffer until warmup threshold, then fit + build.
        if not self.reducer.is_fitted:
            self._warmup_buf.append(item)
            if len(self._warmup_buf) >= self.config.pca_warmup:
                self._fit_and_build()
            return

        # PCA, fitted: project and insert.
        item.vector = self._route_vector(item.raw)
        self._insert(item)
        if self.config.pca_mode == "incremental":
            self._pf_buf.append(item.raw)
            if len(self._pf_buf) >= self.config.pca_batch_size:
                self.reducer.partial_fit(np.stack(self._pf_buf))
                self._pf_buf = []

    def _fit_and_build(self) -> None:
        buf = self._warmup_buf
        self.reducer.fit(np.stack([_raw_of(it) for it in buf]))
        self._warmup_buf = []
        self._reset_tree()
        self._bulk_insert(buf)

    def _bulk_insert(self, items: list[_StoredContent]) -> None:
        for it in items:
            it.vector = self._route_vector(_raw_of(it))
            it.raw = _raw_of(it)  # ensure raw retained for future rebalance
            self._insert(it)

    # --------------------------------------------------------------- routing

    def _route(self, vec: Vector) -> list[_TreeNode]:
        """Descend root->leaf by nearest centroid; return the path of nodes."""
        node = self.root
        path = [node]
        while not node.is_leaf:
            node = min(node.children, key=lambda c: _dist(vec, c.centroid))  # type: ignore[arg-type]
            path.append(node)
        return path

    def _insert(self, item: _StoredContent) -> None:
        path = self._route(item.vector)
        for node in path:  # update centroids along the path
            # vsum starts empty (dim unknown until the first vector arrives)
            node.vsum = item.vector.copy() if node.count == 0 else node.vsum + item.vector
            node.count += 1
        leaf = path[-1]
        leaf.items.append(item)  # type: ignore[union-attr]
        if not leaf.unsplittable and len(leaf.items) > self.config.leaf_capacity:  # type: ignore[arg-type]
            self._split(leaf)

    # --------------------------------------------------------------- splitting

    def _split(self, leaf: _TreeNode) -> None:
        items = leaf.items or []
        decision = self._splitter.split(items)
        if not decision.should_split:
            leaf.unsplittable = decision.reason == "unsplittable"
            return

        children: list[_TreeNode] = []
        for bucket in decision.clusters:
            child = self._new_node()
            child.items = bucket
            child.vsum = np.sum([it.vector for it in bucket], axis=0)
            child.count = len(bucket)
            children.append(child)

        leaf.items = None
        leaf.children = children
        # KMeans can produce uneven clusters; keep splitting oversized children.
        for child in children:
            if len(child.items) > self.config.leaf_capacity:  # type: ignore[arg-type]
                self._split(child)

    # ----------------------------------------------------------------- helpers

    def _in_warmup(self) -> bool:
        return self.reducer.kind != "identity" and not self.reducer.is_fitted

    def _prep_raw(self, vec: Vector) -> Vector:
        """Normalize the raw embedding (so warmup-space search is cosine too)."""
        return _normalize(np.asarray(vec, dtype=np.float64).ravel())

    def _route_vector(self, raw: Vector) -> Vector:
        """Raw embedding -> normalized routing vector (PCA-reduced if active)."""
        if self.reducer.kind == "identity":
            return self._prep_routing(raw)
        reduced = self.reducer.transform(raw[None, :])[0]
        return self._prep_routing(reduced)

    def _prep_routing(self, vec: Vector) -> Vector:
        return _normalize(np.asarray(vec, dtype=np.float64).ravel())

    def _rank(self, q: Vector, items: list[_StoredContent], k: int) -> list[tuple[Hashable, float, Any]]:
        scored = sorted(((_dist(q, it.vector), it) for it in items), key=lambda t: t[0])
        return [(it.id, float(d), it.metadata) for d, it in scored[:k]]

    def _all_items(self) -> list[_StoredContent]:
        return self._node_items(self.root)

    def _node_items(self, node: _TreeNode) -> list[_StoredContent]:
        out: list[_StoredContent] = []
        stack = [node]
        while stack:
            n = stack.pop()
            if n.is_leaf:
                out.extend(n.items or [])
            else:
                stack.extend(n.children or [])
        return out

    def _reset_tree(self) -> None:
        self._next_node_id = 0
        self.root = self._new_node()
        self.root.items = []

    def _new_node(self) -> _TreeNode:
        node = _TreeNode(id=self._next_node_id, vsum=np.zeros(0), count=0)
        self._next_node_id += 1
        return node

    def _take_item_id(self, item_id: Hashable | None) -> Hashable:
        if item_id is None:
            iid = self._next_item_id
            self._next_item_id += 1
            return iid
        if isinstance(item_id, int):
            self._next_item_id = max(self._next_item_id, item_id + 1)
        return item_id

    # --------------------------------------------------------- (de)serialize

    def dump(self) -> MaterializedTreeState:
        return {
            "version": 2,
            "config": {"pca_mode": self.config.pca_mode},
            "next_item_id": self._next_item_id,
            "inserts_since_rebalance": self._inserts_since_rebalance,
            "reducer": self.reducer.to_dict(),
            "warmup_buf": [_item_to_dict(it) for it in self._warmup_buf],
            "pf_buf": [v.tolist() for v in self._pf_buf],
            "root": _node_to_dict(self.root),
        }

    def _restore(self, state: MaterializedTreeState) -> None:
        self._next_item_id = state.get("next_item_id", 0)
        self._inserts_since_rebalance = state.get("inserts_since_rebalance", 0)
        if "reducer" in state:
            self.reducer = Reducer.from_dict(state["reducer"])
        self._warmup_buf = [_dict_to_item(d) for d in state.get("warmup_buf", [])]
        self._pf_buf = [np.asarray(v, dtype=np.float64) for v in state.get("pf_buf", [])]
        self._next_node_id = 0
        self.root = self._dict_to_node(state["root"])

    def _load_state(self) -> MaterializedTreeState | None:
        if self.state is None or not hasattr(self.state, "load"):
            return None
        return self.state.load()

    def _save_state(self) -> None:
        if self.state is not None and hasattr(self.state, "save"):
            self.state.save(self.dump())

    def _dict_to_node(self, d: dict) -> _TreeNode:
        node = self._new_node()
        node.vsum = np.asarray(d["vsum"], dtype=np.float64)
        node.count = d["count"]
        node.unsplittable = d.get("unsplittable", False)
        node.label = d.get("label")
        if d.get("children") is not None:
            node.children = [self._dict_to_node(c) for c in d["children"]]
        else:
            node.items = [_dict_to_item(it) for it in d["items"]]
        return node


def _item_to_dict(it: _StoredContent) -> dict:
    return {
        "id": it.id,
        "vector": it.vector.tolist(),
        "metadata": it.metadata,
        "raw": None if it.raw is None else it.raw.tolist(),
        "text": it.text,
    }


def _dict_to_item(d: dict) -> _StoredContent:
    raw = d.get("raw")
    return _StoredContent(
        id=d["id"],
        vector=np.asarray(d["vector"], dtype=np.float64),
        metadata=d.get("metadata"),
        raw=None if raw is None else np.asarray(raw, dtype=np.float64),
        text=d.get("text"),
    )


def _node_to_dict(node: _TreeNode) -> dict:
    d: dict[str, Any] = {
        "vsum": node.vsum.tolist(),
        "count": node.count,
        "unsplittable": node.unsplittable,
        "label": node.label,
    }
    if node.is_leaf:
        d["children"] = None
        d["items"] = [_item_to_dict(it) for it in (node.items or [])]
    else:
        d["children"] = [_node_to_dict(c) for c in (node.children or [])]
        d["items"] = None
    return d


def _short(text: str, n: int = 60) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _normalize(v: Vector) -> Vector:
    """L2-normalize (cosine semantics); zero vectors are left unchanged."""
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _dist(a: Vector, b: Vector) -> float:
    """Euclidean distance over already-normalized vectors (== cosine ranking)."""
    return float(np.linalg.norm(a - b))


def _content_leaves(branch: BranchNode) -> list[ContentNode]:
    leaves: list[ContentNode] = []
    stack: list[BranchNode | ContentNode] = [branch]
    while stack:
        node = stack.pop()
        if isinstance(node, ContentNode):
            leaves.append(node)
        else:
            stack.extend(reversed(node.children))
    return leaves
