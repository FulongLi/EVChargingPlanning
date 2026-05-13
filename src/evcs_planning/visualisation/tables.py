"""Result table helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_table_csv(table: pd.DataFrame, path: str | Path) -> None:
    """Write a result table as CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
