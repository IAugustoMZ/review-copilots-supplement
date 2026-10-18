"""Reflected table access.

Table structure is defined once, in src/db/migrations/*.sql -- this module gets
SQLAlchemy Core Table objects by reflecting the live database rather than
maintaining a second, parallel schema definition that could drift from the SQL.
"""

from __future__ import annotations

from sqlalchemy import Engine, MetaData, Table


def reflect_table(engine: Engine, name: str, metadata: MetaData | None = None) -> Table:
    metadata = metadata or MetaData()
    return Table(name, metadata, autoload_with=engine)
