"""Trial registry: an append-only log of every experiment run (reports/registry.jsonl).

Its main job is honest trial counting. The deflated Sharpe ratio needs the number of variants
that were tried, and that includes variants tried in earlier runs of the same idea. Re-running
an experiment with a tweaked grid until something passes is the most common way to overfit;
the registry makes those earlier attempts count. Trials are identified by strategy, parameters,
universe and bar interval, so re-running an identical configuration does not double-count.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from tiq.paths import PROJECT_ROOT, reports_dir
from tiq.research.config import canonical_json


def registry_path() -> Path:
    return reports_dir() / "registry.jsonl"


def trial_id(strategy: str, params: dict[str, Any], universe: dict[str, Any], interval: str) -> str:
    payload = canonical_json({"s": strategy, "p": params, "u": universe, "i": interval})
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def read_registry(path: Path | None = None) -> list[dict[str, Any]]:
    path = path or registry_path()
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def family_trial_ids(family: str, path: Path | None = None) -> set[str]:
    ids: set[str] = set()
    for entry in read_registry(path):
        if entry.get("family") == family:
            ids.update(entry.get("trial_ids", []))
    return ids


def append_entry(entry: dict[str, Any], path: Path | None = None) -> None:
    path = path or registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(entry, default=str) + "\n")


def git_state() -> dict[str, Any]:
    """Commit hash and whether the working tree had uncommitted changes, for reproducibility."""
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True,
                              timeout=10).stdout.strip()
    try:
        return {"commit": git("rev-parse", "--short", "HEAD") or None, "dirty": bool(git("status", "--porcelain"))}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}
