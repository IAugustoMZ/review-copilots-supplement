from __future__ import annotations

import gzip
import sqlite3
from pathlib import Path

import pytest
from typer.testing import CliRunner

from review_pipeline.src.cli import db_commands
from review_pipeline.src.cli.app import app

runner = CliRunner()


@pytest.fixture
def db_at(tmp_path: Path, monkeypatch):
    """Point the snapshot/restore commands at a throwaway DB path."""
    live = tmp_path / "review.db"
    snap = tmp_path / "review.db.gz"
    monkeypatch.setattr(db_commands, "DEFAULT_DB_PATH", live)
    monkeypatch.setattr(db_commands, "DEFAULT_SNAPSHOT_PATH", snap)
    return live, snap


def _make_db(path: Path, rows: int = 5) -> None:
    """Rebuilds from scratch, so tests can call it twice to model a live DB that
    moved on after a snapshot was taken."""
    path.unlink(missing_ok=True)
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE classifications_human (record_id TEXT, label TEXT)")
    con.executemany(
        "INSERT INTO classifications_human VALUES (?, ?)",
        [(f"r{i}", "include") for i in range(rows)],
    )
    con.commit()
    con.close()


def test_snapshot_round_trips_the_human_classifications(db_at, tmp_path: Path) -> None:
    live, snap = db_at
    _make_db(live, rows=7)

    assert runner.invoke(app, ["db", "snapshot", "--output", str(snap)]).exit_code == 0
    assert snap.exists()

    live.unlink()  # simulate the loss this whole mechanism exists for
    assert runner.invoke(app, ["db", "restore", "--source", str(snap)]).exit_code == 0

    con = sqlite3.connect(live)
    assert con.execute("SELECT COUNT(*) FROM classifications_human").fetchone()[0] == 7
    con.close()


def test_snapshot_is_actually_compressed_and_smaller(db_at) -> None:
    live, snap = db_at
    _make_db(live, rows=500)
    runner.invoke(app, ["db", "snapshot", "--output", str(snap)])

    assert snap.stat().st_size < live.stat().st_size
    with gzip.open(snap, "rb") as fh:
        assert fh.read(16).startswith(b"SQLite format 3")


def test_restore_refuses_to_clobber_an_existing_database(db_at) -> None:
    """The dangerous direction: restoring over a live DB would discard work that
    may be newer than the snapshot."""
    live, snap = db_at
    _make_db(live, rows=3)
    runner.invoke(app, ["db", "snapshot", "--output", str(snap)])
    _make_db(live, rows=99)  # live has since moved on

    result = runner.invoke(app, ["db", "restore", "--source", str(snap)])
    assert result.exit_code == 1
    assert "already exists" in result.stdout

    con = sqlite3.connect(live)
    assert con.execute("SELECT COUNT(*) FROM classifications_human").fetchone()[0] == 99
    con.close()


def test_restore_overwrites_when_forced(db_at) -> None:
    live, snap = db_at
    _make_db(live, rows=3)
    runner.invoke(app, ["db", "snapshot", "--output", str(snap)])
    _make_db(live, rows=99)

    assert runner.invoke(app, ["db", "restore", "--source", str(snap), "--force"]).exit_code == 0
    con = sqlite3.connect(live)
    assert con.execute("SELECT COUNT(*) FROM classifications_human").fetchone()[0] == 3
    con.close()


def test_snapshot_without_a_database_fails_loudly(db_at) -> None:
    _live, snap = db_at
    result = runner.invoke(app, ["db", "snapshot", "--output", str(snap)])
    assert result.exit_code == 1
    assert not snap.exists()


def test_restore_from_a_missing_snapshot_fails_loudly(db_at) -> None:
    _live, snap = db_at
    result = runner.invoke(app, ["db", "restore", "--source", str(snap)])
    assert result.exit_code == 1


def test_interrupted_snapshot_leaves_no_partial_file_in_place(db_at, monkeypatch) -> None:
    """A truncated snapshot that a later restore would trust is the one failure
    mode worse than no snapshot at all."""
    live, snap = db_at
    _make_db(live, rows=10)

    def boom(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(db_commands.shutil, "copyfileobj", boom)
    with pytest.raises(OSError):
        runner.invoke(app, ["db", "snapshot", "--output", str(snap)], catch_exceptions=False)

    assert not snap.exists()
