"""Demand model entry points."""

from __future__ import annotations

import pandas as pd

from evcs_planning.demand.features import build_rule_based_demand


def prepare_demand_points(demand_points: pd.DataFrame) -> pd.DataFrame:
    """Prepare demand points for clustering and optimisation."""
    required = {"lon", "lat", "residential_score", "commercial_score", "road_score", "accessibility", "grid_risk"}
    missing = required.difference(demand_points.columns)
    if missing:
        raise ValueError(f"Demand points missing required columns: {sorted(missing)}")
    return build_rule_based_demand(demand_points)
