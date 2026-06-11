"""Compatibility import for SQLAlchemy tree cache."""

from embed_tree.loaders.sqlalchemy import SQLAlchemyTreeLoader


class SQLAlchemyTreeCache(SQLAlchemyTreeLoader):
    """Backward-compatible name for SQLAlchemyTreeLoader."""
