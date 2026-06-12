"""SQLAlchemy external persister."""

from __future__ import annotations

from typing import Any

from embed_tree.representation import BranchNode, ContentNode


class SQLAlchemyTreePersister:
    """Persist ContentNode leaves from a BranchNode to a SQL table."""

    def __init__(
        self,
        engine_or_url: Any,
        table_name: str,
        *,
        id_column: str = "id",
        text_column: str = "text",
        metadata_column: str = "metadata",
    ) -> None:
        self.engine_or_url = engine_or_url
        self.table_name = table_name
        self.id_column = id_column
        self.text_column = text_column
        self.metadata_column = metadata_column

    def save(self, state: BranchNode) -> None:
        if not isinstance(state, BranchNode):
            raise TypeError("SQLAlchemyTreePersister persists BranchNode instances")
        sa = _sqlalchemy()
        engine = _engine(sa, self.engine_or_url)
        table = self._table(sa)
        table.metadata.create_all(engine)
        rows = [
            {
                self.id_column: node.id,
                self.text_column: node.text,
                self.metadata_column: node.metadata,
            }
            for node in _content_leaves(state)
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
            sa.Column(self.text_column, sa.Text, nullable=False),
            sa.Column(self.metadata_column, sa.JSON),
        )


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
