"""Simplified location-capacity optimisation solver.

The current implementation uses a transparent greedy heuristic instead of a full
MILP solver so it can run without external optimisation dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from evcs_planning.data.spatial import pairwise_haversine_km
from evcs_planning.optimisation.candidate_generation import assign_candidates_to_clusters
from evcs_planning.optimisation.constraints import validate_optimisation_inputs


@dataclass(frozen=True)
class OptimisationParams:
    max_stations_per_cluster: int = 3
    charger_capacity: float = 45.0
    max_chargers_per_station: int = 10
    coverage_radius_km: float = 2.4
    distance_cost_weight: float = 1.0
    investment_cost_weight: float = 0.00008
    grid_cost_weight: float = 0.22
    capacity_penalty_weight: float = 1.3
    charger_power_kw: float = 50.0


def params_from_mapping(config: dict[str, Any]) -> OptimisationParams:
    """Build solver parameters from YAML config."""
    data = config.get("optimisation", {})
    return OptimisationParams(
        max_stations_per_cluster=int(data.get("max_stations_per_cluster", 3)),
        charger_capacity=float(data.get("charger_capacity", 45.0)),
        max_chargers_per_station=int(data.get("max_chargers_per_station", 10)),
        coverage_radius_km=float(data.get("coverage_radius_km", 2.4)),
        distance_cost_weight=float(data.get("distance_cost_weight", 1.0)),
        investment_cost_weight=float(data.get("investment_cost_weight", 0.00008)),
        grid_cost_weight=float(data.get("grid_cost_weight", 0.22)),
        capacity_penalty_weight=float(data.get("capacity_penalty_weight", 1.3)),
        charger_power_kw=float(data.get("charger_power_kw", 50.0)),
    )


def solve_clustered_planning(
    demand_points: pd.DataFrame,
    candidates: pd.DataFrame,
    labels,
    params: OptimisationParams,
    method_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select charging sites and assign demand points within each cluster."""
    validate_optimisation_inputs(demand_points, candidates)
    demand = demand_points.copy()
    demand["cluster"] = np.asarray(labels, dtype=int)
    clustered_candidates = assign_candidates_to_clusters(candidates, demand, labels)

    selected_frames: list[pd.DataFrame] = []
    assignment_frames: list[pd.DataFrame] = []

    for cluster in sorted(demand["cluster"].unique()):
        cluster_demand = demand[demand["cluster"] == cluster].reset_index(drop=True)
        cluster_candidates = clustered_candidates[clustered_candidates["cluster"] == cluster].reset_index(drop=True)
        if cluster_candidates.empty:
            cluster_candidates = candidates.copy().reset_index(drop=True)
        selected = _select_sites_for_cluster(cluster_demand, cluster_candidates, params)
        assignments = _assign_demand_to_selected_sites(cluster_demand, selected, params)
        selected_frames.append(selected)
        assignment_frames.append(assignments)

    selected_sites = pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame()
    assignments = pd.concat(assignment_frames, ignore_index=True) if assignment_frames else pd.DataFrame()
    selected_sites["method"] = method_name
    assignments["method"] = method_name
    return selected_sites, assignments


def solve_preselected_sites(
    demand_points: pd.DataFrame,
    selected_candidates: pd.DataFrame,
    params: OptimisationParams,
    method_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate a baseline that directly selects a fixed set of candidate sites."""
    validate_optimisation_inputs(demand_points, selected_candidates)
    demand = demand_points.copy()
    demand["cluster"] = 0
    selected = selected_candidates.copy().reset_index(drop=True)
    selected["cluster"] = 0
    selected["selected"] = 1
    selected["assigned_demand"] = 0.0
    selected["chargers"] = 1
    assignments = _assign_demand_to_selected_sites(demand, selected, params)
    selected["method"] = method_name
    assignments["method"] = method_name
    return selected, assignments


def _select_sites_for_cluster(demand: pd.DataFrame, candidates: pd.DataFrame, params: OptimisationParams) -> pd.DataFrame:
    total_demand = float(demand["demand"].sum())
    nominal_station_capacity = params.charger_capacity * params.max_chargers_per_station
    target_count = max(1, int(np.ceil(total_demand / max(nominal_station_capacity, 1.0))) + 1)
    target_count = min(target_count, params.max_stations_per_cluster, len(candidates))

    distances = pairwise_haversine_km(
        demand["lon"].to_numpy(),
        demand["lat"].to_numpy(),
        candidates["lon"].to_numpy(),
        candidates["lat"].to_numpy(),
    )
    within_radius = np.maximum(0.0, params.coverage_radius_km - distances)
    coverage_value = (within_radius * demand["demand"].to_numpy()[:, None]).sum(axis=0)

    cost = (
        params.investment_cost_weight * candidates["fixed_cost"].to_numpy()
        + 8.0 * params.grid_cost_weight * candidates["grid_risk"].to_numpy()
        + 0.35 * np.maximum(0.0, 1.0 - candidates.get("accessibility", 0.5).to_numpy())
    )
    score = coverage_value / np.maximum(cost, 1e-6)
    selected_index = np.argsort(score)[::-1][:target_count]
    selected = candidates.iloc[selected_index].copy().reset_index(drop=True)
    selected["selected"] = 1
    selected["assigned_demand"] = 0.0
    selected["chargers"] = 1
    return selected


def _assign_demand_to_selected_sites(demand: pd.DataFrame, selected: pd.DataFrame, params: OptimisationParams) -> pd.DataFrame:
    distances = pairwise_haversine_km(
        demand["lon"].to_numpy(),
        demand["lat"].to_numpy(),
        selected["lon"].to_numpy(),
        selected["lat"].to_numpy(),
    )
    nearest = distances.argmin(axis=1)
    nearest_distance = distances[np.arange(len(demand)), nearest]

    assignments = demand[["demand_id", "cluster", "demand", "lon", "lat"]].copy()
    assignments["site_id"] = selected.iloc[nearest]["site_id"].to_numpy()
    assignments["service_distance_km"] = nearest_distance
    assignments["covered"] = nearest_distance <= params.coverage_radius_km

    assigned = assignments.groupby("site_id")["demand"].sum()
    selected.loc[:, "assigned_demand"] = selected["site_id"].map(assigned).fillna(0.0).to_numpy()
    selected.loc[:, "chargers"] = np.ceil(selected["assigned_demand"] / params.charger_capacity).clip(
        lower=1,
        upper=np.minimum(params.max_chargers_per_station, selected["max_chargers"].to_numpy()),
    ).astype(int)
    selected.loc[:, "station_capacity"] = selected["chargers"] * params.charger_capacity
    selected.loc[:, "capacity_slack"] = selected["station_capacity"] - selected["assigned_demand"]
    return assignments
