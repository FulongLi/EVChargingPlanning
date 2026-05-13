"""Configuration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in YAML file: {path}")
    return data


def ensure_dirs(paths: Mapping[str, str | Path] | list[str | Path]) -> None:
    """Create output directories if they do not already exist."""
    values = paths.values() if isinstance(paths, Mapping) else paths
    for path in values:
        Path(path).mkdir(parents=True, exist_ok=True)
