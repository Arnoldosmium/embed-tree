"""SQLite-specific tree loader."""

from __future__ import annotations

from pathlib import Path

from .sqlalchemy import SQLAlchemyTreeLoader, _sqlalchemy


class SQLiteTreeLoader(SQLAlchemyTreeLoader):
    """SQLite tree-state loader with built-in table creation."""

    def __init__(self, path: str | Path, *, table_name: str = "embed_tree_state", cache_key: str = "default") -> None:
        self.path = Path(path)
        super().__init__(f"sqlite:///{self.path}", table_name=table_name, cache_key=cache_key)

    def post_init(self) -> None:
        sa = _sqlalchemy()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._table(sa).metadata.create_all(self.engine)

