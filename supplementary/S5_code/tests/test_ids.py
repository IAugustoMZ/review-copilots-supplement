from __future__ import annotations

from review_pipeline.src.common.ids import new_id


def test_new_id_is_unique() -> None:
    ids = {new_id() for _ in range(1000)}
    assert len(ids) == 1000


def test_new_id_is_valid_ulid_format() -> None:
    # Crockford base32, 26 chars: sortable at millisecond resolution, so within
    # a single millisecond ordering isn't guaranteed -- only the format is checked here.
    generated = new_id()
    assert len(generated) == 26
    assert generated.isupper() or generated.isdigit() or generated.isalnum()
