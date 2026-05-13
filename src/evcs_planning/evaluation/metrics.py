"""Evaluation metrics for EV charging station planning results."""

from __future__ import annotations

import numpy as np
import pandas as pd

from evcs_planning.optimisation.objective import station_grid_impact, station_investment_cost
from evcs_planning.optimisation.solver import OptimisationParams


def evaluate_solution(
    selected_sites: pd.DataFrame,
    assignments: pd.DataFrame,
    params: OptimisationParams,
) -> dict[str, float]:
    """Compute the main comparison metrics for one planning method."""
    total_demand = float(assignments["demand"].sum()) if not assignments.empty else 0.0
    covered_demand = float(assignments.loc[assignments["covered"], "demand"].sum()) if not assignments.empty else 0.0
    weighted_distance = (
        float((assignments["service_distance_km"] * assignments["demand"]).sum()) / total_demand
        if total_demand > 0.0
        else 0.0
    )
    utilisation = selected_sites["assigned_demand"] / selected_sites["station_capacity"].replace(0, np.nan)
    utilisation = utilisation.fillna(0.0)
    excess = np.maximum(0.0, selected_sites["assigned_demand"] - selected_sites["station_capacity"])

    grid_impact = station_grid_impact(selected_sites, charger_power_kw=params.charger_power_kw)
    installed_chargers = float(selected_sites["chargers"].sum()) if not selected_sites.empty else 0.0

    return {
        "average_service_distance_km": weighted_distance,
        "demand_coverage_ratio": covered_demand / total_demand if total_demand > 0.0 else 0.0,
        "total_investment_cost": station_investment_cost(selected_sites),
        "grid_impact_score": grid_impact,
        "grid_impact_per_charger": grid_impact / installed_chargers if installed_chargers > 0.0 else 0.0,
        "station_utilisation_mean": float(utilisation.mean()) if len(utilisation) else 0.0,
        "station_utilisation_std": float(utilisation.std(ddof=0)) if len(utilisation) else 0.0,
        "unserved_demand": float(excess.sum()),
        "selected_station_count": float(len(selected_sites)),
        "installed_chargers": installed_chargers,
    }


def comparison_table(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Convert metric dictionaries to a tidy comparison table."""
    rows = []
    for method, metrics in results.items():
        row = {"method": method}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)
