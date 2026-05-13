"""Demand feature construction."""

from __future__ import annotations

import pandas as pd

from evcs_planning.data.spatial import min_max_scale


def build_rule_based_demand(
    frame: pd.DataFrame,
    residential_weight: float = 0.45,
    commercial_weight: float = 0.35,
    road_weight: float = 0.20,
) -> pd.DataFrame:
    """Build a normalised total demand score from three interpretable components."""
    result = frame.copy()
    raw = (
        residential_weight * result["residential_score"].to_numpy()
        + commercial_weight * result["commercial_score"].to_numpy()
        + road_weight * result["road_score"].to_numpy()
    )
    result["demand_score"] = min_max_scale(raw)
    if "demand" not in result.columns:
        result["demand"] = 20.0 + 60.0 * result["demand_score"]
    return result
