"""Data loading utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV file with a clear error if it is missing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    return pd.read_csv(path)


def save_csv(frame: pd.DataFrame, path: str | Path) -> None:
    """Save a DataFrame to CSV, creating the parent folder."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
