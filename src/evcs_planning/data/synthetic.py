"""Synthetic London-like data generation for the MVP pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from evcs_planning.data.spatial import min_max_scale


@dataclass(frozen=True)
class Bounds:
    lon_min: float = -0.34
    lon_max: float = -0.02
    lat_min: float = 51.48
    lat_max: float = 51.65


@dataclass(frozen=True)
class SyntheticConfig:
    n_demand_points: int = 420
    n_candidate_sites: int = 95
    n_existing_stations: int = 28
    random_seed: int = 42
    bounds: Bounds = Bounds()


def config_from_mapping(config: dict[str, Any]) -> SyntheticConfig:
    """Build a synthetic config from a nested YAML mapping."""
    synthetic = config.get("synthetic", {})
    bounds_data = synthetic.get("bounds", {})
    bounds = Bounds(
        lon_min=float(bounds_data.get("lon_min", Bounds.lon_min)),
        lon_max=float(bounds_data.get("lon_max", Bounds.lon_max)),
        lat_min=float(bounds_data.get("lat_min", Bounds.lat_min)),
        lat_max=float(bounds_data.get("lat_max", Bounds.lat_max)),
    )
    return SyntheticConfig(
        n_demand_points=int(synthetic.get("n_demand_points", 420)),
        n_candidate_sites=int(synthetic.get("n_candidate_sites", 95)),
        n_existing_stations=int(synthetic.get("n_existing_stations", 28)),
        random_seed=int(config.get("random_seed", synthetic.get("random_seed", 42))),
        bounds=bounds,
    )


def generate_synthetic_case(config: SyntheticConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate demand points, candidate sites, and existing stations."""
    rng = np.random.default_rng(config.random_seed)
    demand = _generate_demand_points(rng, config)
    candidates = _generate_candidate_sites(rng, config, demand)
    existing = _generate_existing_stations(rng, config, demand)
    return demand, candidates, existing


def _sample_urban_mixture(rng: np.random.Generator, n_points: int, bounds: Bounds) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centers = np.array(
        [
            [-0.20, 51.54],
            [-0.12, 51.52],
            [-0.27, 51.60],
            [-0.08, 51.59],
        ]
    )
    probabilities = np.array([0.34, 0.26, 0.23, 0.17])
    component = rng.choice(len(centers), size=n_points, p=probabilities)
    lon = centers[component, 0] + rng.normal(0.0, 0.030, size=n_points)
    lat = centers[component, 1] + rng.normal(0.0, 0.020, size=n_points)
    lon = np.clip(lon, bounds.lon_min, bounds.lon_max)
    lat = np.clip(lat, bounds.lat_min, bounds.lat_max)
    return lon, lat, component


def _generate_demand_points(rng: np.random.Generator, config: SyntheticConfig) -> pd.DataFrame:
    lon, lat, component = _sample_urban_mixture(rng, config.n_demand_points, config.bounds)
    westness = min_max_scale(config.bounds.lon_max - lon)
    centrality = 1.0 - min_max_scale(np.hypot(lon + 0.16, lat - 51.55))
    road_corridor = np.exp(-((lat - 51.56) ** 2) / 0.00045)

    residential = 0.45 + 0.55 * westness + rng.normal(0.0, 0.08, size=config.n_demand_points)
    commercial = 0.35 + 0.65 * centrality + rng.normal(0.0, 0.08, size=config.n_demand_points)
    road = 0.25 + 0.75 * road_corridor + rng.normal(0.0, 0.08, size=config.n_demand_points)

    residential = min_max_scale(residential)
    commercial = min_max_scale(commercial)
    road = min_max_scale(road)

    accessibility = min_max_scale(0.55 * road + 0.45 * centrality + rng.normal(0.0, 0.06, size=config.n_demand_points))
    hosting_capacity = min_max_scale(1.15 - 0.55 * centrality - 0.25 * commercial + rng.normal(0.0, 0.08, size=config.n_demand_points), 0.25, 1.0)
    peak_proxy = min_max_scale(0.5 * residential + 0.35 * commercial + 0.15 * road, 0.2, 1.1)
    grid_risk = min_max_scale(peak_proxy / hosting_capacity)

    demand = 6.0 + 28.0 * min_max_scale(0.45 * residential + 0.35 * commercial + 0.20 * road)
    demand = np.maximum(3.0, demand + rng.normal(0.0, 1.5, size=config.n_demand_points))

    return pd.DataFrame(
        {
            "demand_id": [f"d{i:04d}" for i in range(config.n_demand_points)],
            "lon": lon,
            "lat": lat,
            "synthetic_zone": component,
            "residential_score": residential,
            "commercial_score": commercial,
            "road_score": road,
            "accessibility": accessibility,
            "hosting_capacity_proxy": hosting_capacity,
            "peak_load_proxy": peak_proxy,
            "grid_risk": grid_risk,
            "demand": demand,
        }
    )


def _generate_candidate_sites(rng: np.random.Generator, config: SyntheticConfig, demand: pd.DataFrame) -> pd.DataFrame:
    sampled = demand.sample(n=min(len(demand), config.n_candidate_sites), random_state=config.random_seed).reset_index(drop=True)
    lon = sampled["lon"].to_numpy() + rng.normal(0.0, 0.008, size=len(sampled))
    lat = sampled["lat"].to_numpy() + rng.normal(0.0, 0.006, size=len(sampled))
    lon = np.clip(lon, config.bounds.lon_min, config.bounds.lon_max)
    lat = np.clip(lat, config.bounds.lat_min, config.bounds.lat_max)

    grid_risk = min_max_scale(sampled["grid_risk"].to_numpy() + rng.normal(0.0, 0.12, size=len(sampled)))
    accessibility = min_max_scale(sampled["accessibility"].to_numpy() + rng.normal(0.0, 0.10, size=len(sampled)))
    fixed_cost = 52000 + 58000 * grid_risk + 18000 * (1.0 - accessibility)
    charger_cost = 8500 + 2200 * grid_risk

    return pd.DataFrame(
        {
            "site_id": [f"c{i:03d}" for i in range(len(sampled))],
            "lon": lon,
            "lat": lat,
            "grid_risk": grid_risk,
            "accessibility": accessibility,
            "fixed_cost": fixed_cost,
            "charger_cost": charger_cost,
            "max_chargers": rng.integers(12, 26, size=len(sampled)),
        }
    )


def _generate_existing_stations(rng: np.random.Generator, config: SyntheticConfig, demand: pd.DataFrame) -> pd.DataFrame:
    sampled = demand.sample(n=min(len(demand), config.n_existing_stations), random_state=config.random_seed + 7).reset_index(drop=True)
    return pd.DataFrame(
        {
            "station_id": [f"e{i:03d}" for i in range(len(sampled))],
            "lon": sampled["lon"].to_numpy() + rng.normal(0.0, 0.006, size=len(sampled)),
            "lat": sampled["lat"].to_numpy() + rng.normal(0.0, 0.004, size=len(sampled)),
            "chargers": rng.integers(2, 8, size=len(sampled)),
        }
    )
