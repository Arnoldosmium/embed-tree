"""SQLAlchemy-backed tree loader."""

from __future__ import annotations

from typing import Any

from embed_tree.persisters.model import MaterializedTreeState
from embed_tree.representation import PartialTree
from embed_tree.representation.default import partial_tree_from_dict, partial_tree_to_dict


class SQLAlchemyTreeLoader:
    """Load/save tree representation from an existing SQL table.

    This base implementation deliberately does not create tables or run
    migrations. Database shape is an application decision. Subclasses can
    override ``post_init`` for setup; ``SQLiteTreeLoader`` is the built-in
    simple implementation that creates its table.
    """

    def __init__(
        self,
        engine_or_url: Any,
        *,
        table_name: str = "embed_tree_state",
        cache_key: str = "default",
    ) -> None:
        self.engine_or_url = engine_or_url
        self.table_name = table_name
        self.cache_key = cache_key
        sa = _sqlalchemy()
        self.engine = _engine(sa, self.engine_or_url)
        self.post_init()

    def post_init(self) -> None:
        """Hook for migrations or validation owned by the caller/subclass."""
        pass

    def load(self) -> PartialTree | MaterializedTreeState | None:
        sa = _sqlalchemy()
        table = self._table(sa)
        stmt = sa.select(table.c.kind, table.c.payload).where(table.c.cache_key == self.cache_key)
        with self.engine.connect() as conn:
            row = conn.execute(stmt).mappings().first()
        if row is None:
            return None
        if row["kind"] == "partial_tree":
            return partial_tree_from_dict(row["payload"])
        if row["kind"] == "materialized_tree_state":
            return row["payload"]
        return row["payload"]

    def save(self, state: PartialTree | MaterializedTreeState) -> None:
        sa = _sqlalchemy()
        table = self._table(sa)
        if isinstance(state, PartialTree):
            kind = "partial_tree"
            payload: dict[str, Any] = partial_tree_to_dict(state)
        else:
            kind = "materialized_tree_state"
            payload = state

        delete = table.delete().where(table.c.cache_key == self.cache_key)
        insert = table.insert().values(cache_key=self.cache_key, kind=kind, payload=payload)
        with self.engine.begin() as conn:
            conn.execute(delete)
            conn.execute(insert)

    def _table(self, sa: Any) -> Any:
        meta = sa.MetaData()
        return sa.Table(
            self.table_name,
            meta,
            sa.Column("cache_key", sa.String, primary_key=True),
            sa.Column("kind", sa.String, nullable=False),
            sa.Column("payload", sa.JSON, nullable=False),
        )


def _sqlalchemy() -> Any:
    try:
        import sqlalchemy as sa
    except ImportError as e:  # pragma: no cover
        raise ImportError('SQLAlchemyTreeLoader needs the "sql" extra: pip install "embed-tree[sql]"') from e
    return sa


def _engine(sa: Any, engine_or_url: Any) -> Any:
    if isinstance(engine_or_url, str):
        return sa.create_engine(engine_or_url)
    return engine_or_url
