"""Config-file resolution.

All methodology parameters live in review_pipeline/config/*.yaml (search queries,
journals, coding schema, inclusion rules, sentinel set, gap claims -- added phase
by phase). This module is deliberately just path resolution + a thin YAML loader;
each module owns parsing its own config file into typed Pydantic models rather than
this module knowing every config file's shape.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[2]  # .../review_pipeline
CONFIG_DIR = PACKAGE_ROOT / "config"
DATA_DIR = PACKAGE_ROOT / "data"
LOGS_DIR = PACKAGE_ROOT / "logs"
OUTPUTS_DIR = PACKAGE_ROOT / "outputs"
DEFAULT_DB_PATH = DATA_DIR / "normalized" / "review.db"


def config_path(filename: str) -> Path:
    """Resolve a config file by name relative to review_pipeline/config/."""
    path = CONFIG_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}. "
            f"Expected it under {CONFIG_DIR} -- see docs/TECHNICAL_IMPLEMENTATION_PLAN.md."
        )
    return path


def load_yaml(filename: str) -> dict[str, Any]:
    """Load and parse a YAML config file by name (see config_path)."""
    with config_path(filename).open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data or {}
