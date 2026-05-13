"""Preprocessing utilities shared by synthetic and real-data cases."""

from __future__ import annotations

import pandas as pd

from evcs_planning.data.spatial import min_max_scale


def normalise_columns(frame: pd.DataFrame, columns: list[str], suffix: str = "_norm") -> pd.DataFrame:
    """Return a copy with selected numeric columns min-max normalised."""
    result = frame.copy()
    for column in columns:
        result[f"{column}{suffix}"] = min_max_scale(result[column].to_numpy())
    return result


def require_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    """Validate that a table contains required columns."""
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")
