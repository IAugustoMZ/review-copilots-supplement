"""Stable, sortable IDs for every entity the pipeline creates.

ULIDs (not UUID4) so that record/event IDs are lexicographically sortable by
creation time -- useful when eyeballing `audit_events` or `records` exports
without a separate timestamp sort, and useful for deterministic-looking test
fixtures (a monotonic sequence is easy to assert against).
"""

from __future__ import annotations

from ulid import ULID


def new_id() -> str:
    """A new ULID as a string. Use for any primary key in the schema."""
    return str(ULID())


def new_record_id() -> str:
    return new_id()


def new_event_id() -> str:
    return new_id()
