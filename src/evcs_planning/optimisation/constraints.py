"""Small validation helpers for optimisation inputs."""

from __future__ import annotations

import pandas as pd


def validate_optimisation_inputs(demand_points: pd.DataFrame, candidates: pd.DataFrame) -> None:
    """Validate required columns for the simplified solver."""
    demand_required = {"demand_id", "lon", "lat", "demand"}
    candidate_required = {"site_id", "lon", "lat", "fixed_cost", "charger_cost", "grid_risk", "max_chargers"}
    missing_demand = demand_required.difference(demand_points.columns)
    missing_candidate = candidate_required.difference(candidates.columns)
    if missing_demand:
        raise ValueError(f"Demand points missing columns: {sorted(missing_demand)}")
    if missing_candidate:
        raise ValueError(f"Candidates missing columns: {sorted(missing_candidate)}")
