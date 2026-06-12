"""embed-tree: a browsable hierarchy over content embeddings."""

from .config import LLMConfig, RebalanceConfig, TreeConfig
from .embedders import (
    BaseTextEmbedder,
    HuggingFaceTextEmbedder,
    OpenAITextEmbedder,
    TagSetEmbedder,
    TextEmbedder,
    embed_texts,
)
from .labelers import (
    FunctionLabeler,
    KeywordLabeler,
    LabelCandidate,
    Labeler,
    LabelRequest,
    LLMLabeler,
)
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
from .reducers import (
    FreezePCAReducer,
    IdentityReducer,
    IncrementalPCAReducer,
    Reducer,
)
from .reconcilers import DefaultTreeReconciler, TreeReconciler
from .representation import (
    BranchNode,
    ContentNode,
    DefaultTreeRepresentation,
    NodeId,
    VectorData,
    tree_from_dict,
    tree_to_dict,
)
from .tree import EmbedTree

__all__ = [
    "EmbedTree",
    "TreeConfig",
    "RebalanceConfig",
    "LLMConfig",
    "TextEmbedder",
    "BaseTextEmbedder",
    "HuggingFaceTextEmbedder",
    "OpenAITextEmbedder",
    "TagSetEmbedder",
    "embed_texts",
    "PCAConfig",
    "PCAProjector",
    "VectorProjector",
    "LabelCandidate",
    "LabelRequest",
    "Labeler",
    "FunctionLabeler",
    "KeywordLabeler",
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
    "BranchNode",
    "ContentNode",
    "NodeId",
    "VectorData",
    "tree_from_dict",
    "tree_to_dict",
    "Reducer",
    "IdentityReducer",
    "FreezePCAReducer",
    "IncrementalPCAReducer",
]
