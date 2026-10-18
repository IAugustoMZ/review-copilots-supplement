"""Structured JSONL run logging (docs/TECHNICAL_IMPLEMENTATION_PLAN.md section 23).

Two audit layers exist in this pipeline: this module (mechanical "what happened"
per operation, e.g. one line per search API call) and the `audit_events` DB table
(durable "what was decided, by whom, superseding what"). Both are append-only --
this module never truncates or overwrites a log file, only appends lines.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from review_pipeline.src.common.config import LOGS_DIR


class RunLogger:
    """Appends one JSON object per line to logs/<component>/<run_id>.jsonl.

    Each call to log_event() writes immediately (not buffered) so a crash mid-run
    still leaves a complete log of everything that happened before it, per the
    "never silently discard a failure" requirement.
    """

    def __init__(self, component: str, run_id: str, logs_dir: Path = LOGS_DIR) -> None:
        self.component = component
        self.run_id = run_id
        self._dir = logs_dir / component
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / f"{run_id}.jsonl"

    @property
    def path(self) -> Path:
        return self._path

    def log_event(self, **fields: Any) -> None:
        record = {"timestamp": datetime.now(UTC).isoformat(), **fields}
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
