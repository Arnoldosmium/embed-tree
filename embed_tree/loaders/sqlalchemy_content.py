"""SQLAlchemy-backed content loader."""

from __future__ import annotations

from typing import Any, Iterable

from embed_tree.representation import ContentNode, PartialTree

from .sqlalchemy import _engine, _sqlalchemy


class SQLAlchemyContentLoader:
    """Load content nodes from an existing SQL table via SQLAlchemy Core."""

    def __init__(
        self,
        engine_or_url: Any,
        table_name: str,
        *,
        id_column: str = "id",
        content_column: str = "content",
        text_column: str | None = None,
        payload_columns: Iterable[str] | None = None,
        where: Any | None = None,
    ) -> None:
        self.engine_or_url = engine_or_url
        self.table_name = table_name
        self.id_column = id_column
        self.content_column = content_column
        self.text_column = text_column
        self.payload_columns = list(payload_columns or [])
        self.where = where
        self.post_init()

    def post_init(self) -> None:
        """Hook for migrations or validation owned by the caller/subclass."""
        pass

    def load(self) -> PartialTree | None:
        sa = _sqlalchemy()
        engine = _engine(sa, self.engine_or_url)
        meta = sa.MetaData()
        table = sa.Table(self.table_name, meta, autoload_with=engine)

        stmt = sa.select(table)
        if self.where is not None:
            stmt = stmt.where(self.where)

        nodes: list[ContentNode] = []
        with engine.connect() as conn:
            for row in conn.execute(stmt).mappings():
                payload = {name: row[name] for name in self.payload_columns}
                nodes.append(
                    ContentNode(
                        id=row[self.id_column],
                        content=row[self.content_column],
                        text=None if self.text_column is None else row[self.text_column],
                        payload=payload or None,
                    )
                )

        return PartialTree(content_nodes=nodes, metadata={"source": "sqlalchemy", "table": self.table_name})

