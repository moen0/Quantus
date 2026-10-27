"""Filesystem locations. Override with the TIQ_DATA_DIR and TIQ_REPORTS_DIR environment variables."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    return Path(os.environ.get("TIQ_DATA_DIR", PROJECT_ROOT / "data"))


def lake_dir() -> Path:
    return data_dir() / "lake"


def reports_dir() -> Path:
    return Path(os.environ.get("TIQ_REPORTS_DIR", PROJECT_ROOT / "reports"))
