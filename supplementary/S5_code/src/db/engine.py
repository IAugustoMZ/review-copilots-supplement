"""SQLite engine + hand-written migration runner.

No ORM migration framework: schema changes are rare and reviewable as plain
numbered .sql files under src/db/migrations/. This module only tracks which
migrations have been applied (schema_migrations table) and applies the rest,
in filename order, inside a single transaction each.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import Engine, create_engine, event

from review_pipeline.src.common.config import DEFAULT_DB_PATH

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"


def get_engine(db_path: Path = DEFAULT_DB_PATH) -> Engine:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}", future=True)

    @event.listens_for(engine, "connect")
    def _enable_foreign_keys(dbapi_connection: sqlite3.Connection, _connection_record: object) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    return engine


def _ensure_migrations_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            migration_id TEXT PRIMARY KEY,
            applied_at   TEXT NOT NULL
        )
        """
    )
    conn.commit()


def applied_migrations(conn: sqlite3.Connection) -> set[str]:
    _ensure_migrations_table(conn)
    rows = conn.execute("SELECT migration_id FROM schema_migrations").fetchall()
    return {row[0] for row in rows}


def pending_migrations(conn: sqlite3.Connection) -> list[Path]:
    already_applied = applied_migrations(conn)
    all_migrations = sorted(MIGRATIONS_DIR.glob("*.sql"))
    return [m for m in all_migrations if m.name not in already_applied]


def run_migrations(engine: Engine) -> list[str]:
    """Apply any migration files not yet recorded in schema_migrations.

    Returns the list of migration filenames that were applied by this call
    (empty if the database was already up to date).
    """
    raw_conn = engine.raw_connection()
    try:
        sqlite_conn: sqlite3.Connection = raw_conn.driver_connection  # type: ignore[attr-defined]
        applied: list[str] = []
        for migration_path in pending_migrations(sqlite_conn):
            sql_text = migration_path.read_text(encoding="utf-8")
            sqlite_conn.executescript(sql_text)
            sqlite_conn.execute(
                "INSERT INTO schema_migrations (migration_id, applied_at) VALUES (?, ?)",
                (migration_path.name, datetime.now(UTC).isoformat()),
            )
            sqlite_conn.commit()
            applied.append(migration_path.name)
        return applied
    finally:
        raw_conn.close()
