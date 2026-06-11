"""Compatibility import for JSON tree cache."""

from embed_tree.loaders.json import JsonTreeLoader


class JsonTreeCache(JsonTreeLoader):
    """Backward-compatible name for JsonTreeLoader."""
