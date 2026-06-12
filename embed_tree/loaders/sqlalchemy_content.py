"""SQLAlchemy-backed content loader."""

from __future__ import annotations

from typing import Any, Iterable

from embed_tree.representation import BranchNode, ContentNode

from .sqlalchemy import _engine, _sqlalchemy


class SQLAlchemyContentLoader:
    """Load content leaves from an existing SQL table."""

    def __init__(
        self,
        engine_or_url: Any,
        table_name: str,
        *,
        id_column: str = "id",
        text_column: str = "text",
        metadata_columns: Iterable[str] | None = None,
        where: Any | None = None,
    ) -> None:
        self.engine_or_url = engine_or_url
        self.table_name = table_name
        self.id_column = id_column
        self.text_column = text_column
        self.metadata_columns = list(metadata_columns or [])
        self.where = where

    def load(self) -> BranchNode | None:
        sa = _sqlalchemy()
        engine = _engine(sa, self.engine_or_url)
        meta = sa.MetaData()
        table = sa.Table(self.table_name, meta, autoload_with=engine)
        stmt = sa.select(table)
        if self.where is not None:
            stmt = stmt.where(self.where)
        children: list[ContentNode] = []
        with engine.connect() as conn:
            for row in conn.execute(stmt).mappings():
                metadata = {name: row[name] for name in self.metadata_columns}
                children.append(ContentNode(id=row[self.id_column], text=row[self.text_column], metadata=metadata))
        return BranchNode(id=self.table_name, label=self.table_name, children=children)
