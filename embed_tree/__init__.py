"""embed-tree: an incremental hierarchical clustering tree over embeddings.

See DESIGN.md for the full design. Minimal usage:

    from embed_tree import EmbedTree, TreeConfig, FileTreeStore

    tree = EmbedTree(embedder=my_embed_fn, store=FileTreeStore("./tree.json"))
    tree.add("some content")
    hits = tree.query("similar content", k=5)
"""

from .config import LLMConfig, RebalanceConfig, TreeConfig
from .embedders import HuggingFaceTextEmbedder, TextEmbedder, embed_texts
from .labelers import FunctionLabeler, LabelCandidate, Labeler, LabelRequest, LLMLabeler
from .loaders import (
    FileSystemTreeLoader,
    JsonTreeLoader,
    SQLAlchemyContentLoader,
    SQLAlchemyTreeLoader,
    SQLiteTreeLoader,
    TreeLoader,
)
from .persisters import (
    FileSystemTreePersister,
    FolderTreePersister,
    JsonTreePersister,
    MaterializedTreeState,
    SQLAlchemyTreePersister,
    TreePersister,
)
from .projectors import PCAConfig, PCAProjector, VectorProjector
from .providers import (
    EmbeddingProvider,
    FakeEmbeddingProvider,
    OpenAIEmbeddingProvider,
    SentenceTransformerProvider,
)
from .reducers import (
    FreezePCAReducer,
    IdentityReducer,
    IncrementalPCAReducer,
    Reducer,
)
from .reconcilers import DefaultTreeReconciler, TreeReconciler
from .representation import (
    ContentNode,
    DefaultTreeRepresentation,
    KeyNode,
    NodeAggregate,
    NodeEmbedding,
    NodeId,
    PartialTree,
    TreeEdge,
    VectorData,
    partial_tree_from_dict,
    partial_tree_to_dict,
)
from .store import FileTreeStore, NullTreeStore, TreeState, TreeStore
from .taggers import KeywordTagger, LLMTagger, Tagger, make_tagger
from .tree import EmbedTree, Item, Node

__all__ = [
    "EmbedTree",
    "Item",
    "Node",
    "TreeConfig",
    "RebalanceConfig",
    "LLMConfig",
    "TextEmbedder",
    "HuggingFaceTextEmbedder",
    "embed_texts",
    "PCAConfig",
    "PCAProjector",
    "VectorProjector",
    "LabelCandidate",
    "LabelRequest",
    "Labeler",
    "FunctionLabeler",
    "LLMLabeler",
    "TreeLoader",
    "FileSystemTreeLoader",
    "JsonTreeLoader",
    "SQLAlchemyContentLoader",
    "SQLAlchemyTreeLoader",
    "SQLiteTreeLoader",
    "MaterializedTreeState",
    "TreeReconciler",
    "DefaultTreeReconciler",
    "TreePersister",
    "FolderTreePersister",
    "FileSystemTreePersister",
    "JsonTreePersister",
    "SQLAlchemyTreePersister",
    "DefaultTreeRepresentation",
    "PartialTree",
    "ContentNode",
    "KeyNode",
    "TreeEdge",
    "NodeEmbedding",
    "NodeAggregate",
    "NodeId",
    "VectorData",
    "partial_tree_from_dict",
    "partial_tree_to_dict",
    "TreeState",
    "TreeStore",
    "FileTreeStore",
    "NullTreeStore",
    "EmbeddingProvider",
    "FakeEmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "SentenceTransformerProvider",
    "Reducer",
    "IdentityReducer",
    "FreezePCAReducer",
    "IncrementalPCAReducer",
    "Tagger",
    "KeywordTagger",
    "LLMTagger",
    "make_tagger",
]
