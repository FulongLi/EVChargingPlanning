"""Candidate site helpers."""

from __future__ import annotations

import pandas as pd


def assign_candidates_to_clusters(candidates: pd.DataFrame, demand_points: pd.DataFrame, labels) -> pd.DataFrame:
    """Assign each candidate to the cluster of its nearest demand point."""
    from evcs_planning.data.spatial import pairwise_haversine_km

    distances = pairwise_haversine_km(
        candidates["lon"].to_numpy(),
        candidates["lat"].to_numpy(),
        demand_points["lon"].to_numpy(),
        demand_points["lat"].to_numpy(),
    )
    nearest_demand = distances.argmin(axis=1)
    result = candidates.copy()
    result["cluster"] = [int(labels[index]) for index in nearest_demand]
    return result
