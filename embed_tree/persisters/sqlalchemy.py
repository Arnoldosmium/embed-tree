"""SQLAlchemy external persister."""

from __future__ import annotations

from typing import Any

from embed_tree.representation import PartialTree


class SQLAlchemyTreePersister:
    """Persist PartialTree content nodes to a SQL table via SQLAlchemy Core."""

    def __init__(
        self,
        engine_or_url: Any,
        table_name: str,
        *,
        id_column: str = "id",
        content_column: str = "content",
        text_column: str = "text",
        payload_column: str = "payload",
    ) -> None:
        self.engine_or_url = engine_or_url
        self.table_name = table_name
        self.id_column = id_column
        self.content_column = content_column
        self.text_column = text_column
        self.payload_column = payload_column

    def save(self, state: Any) -> None:
        if not isinstance(state, PartialTree):
            raise TypeError("SQLAlchemyTreePersister only persists PartialTree instances")

        sa = _sqlalchemy()
        engine = _engine(sa, self.engine_or_url)
        table = self._table(sa)
        table.metadata.create_all(engine)
        rows = [
            {
                self.id_column: node.id,
                self.content_column: node.content,
                self.text_column: node.text,
                self.payload_column: node.payload,
            }
            for node in state.content_nodes
        ]
        with engine.begin() as conn:
            conn.execute(table.delete())
            if rows:
                conn.execute(table.insert(), rows)

    def _table(self, sa: Any) -> Any:
        meta = sa.MetaData()
        return sa.Table(
            self.table_name,
            meta,
            sa.Column(self.id_column, sa.String, primary_key=True),
            sa.Column(self.content_column, sa.Text, nullable=False),
            sa.Column(self.text_column, sa.Text),
            sa.Column(self.payload_column, sa.JSON),
        )


def _sqlalchemy() -> Any:
    try:
        import sqlalchemy as sa
    except ImportError as e:  # pragma: no cover
        raise ImportError('SQLAlchemyTreePersister needs the "sql" extra: pip install "embed-tree[sql]"') from e
    return sa


def _engine(sa: Any, engine_or_url: Any) -> Any:
    if isinstance(engine_or_url, str):
        return sa.create_engine(engine_or_url)
    return engine_or_url

