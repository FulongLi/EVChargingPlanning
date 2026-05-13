"""London/Barnet open-data interface.

The first real-data pipeline uses OpenStreetMap data through Overpass. It keeps
the data model compatible with the synthetic MVP so the same clustering and
optimisation modules can be reused.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from urllib import parse, request

import numpy as np
import pandas as pd

from evcs_planning.data.spatial import min_max_scale, pairwise_haversine_km


DEFAULT_OVERPASS_ENDPOINT = "https://overpass-api.de/api/interpreter"


def fetch_barnet_osm_raw(
    bbox: dict[str, float],
    output_path: str | Path,
    endpoint: str = DEFAULT_OVERPASS_ENDPOINT,
    timeout_seconds: int = 180,
) -> dict[str, Any]:
    """Download OSM features for a Barnet bounding box using Overpass."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    south, west, north, east = bbox["south"], bbox["west"], bbox["north"], bbox["east"]
    bbox_text = f"{south},{west},{north},{east}"
    query = f"""
    [out:json][timeout:{timeout_seconds}];
    (
      node["amenity"="charging_station"]({bbox_text});
      way["amenity"="charging_station"]({bbox_text});
      relation["amenity"="charging_station"]({bbox_text});

      node["amenity"="parking"]({bbox_text});
      way["amenity"="parking"]({bbox_text});
      relation["amenity"="parking"]({bbox_text});

      node["shop"]({bbox_text});
      way["shop"]({bbox_text});
      relation["shop"]({bbox_text});
      node["office"]({bbox_text});
      way["office"]({bbox_text});
      relation["office"]({bbox_text});
      node["amenity"~"^(restaurant|cafe|pub|bar|fast_food|school|college|university|hospital|clinic|library|cinema|theatre|community_centre)$"]({bbox_text});
      way["amenity"~"^(restaurant|cafe|pub|bar|fast_food|school|college|university|hospital|clinic|library|cinema|theatre|community_centre)$"]({bbox_text});
      relation["amenity"~"^(restaurant|cafe|pub|bar|fast_food|school|college|university|hospital|clinic|library|cinema|theatre|community_centre)$"]({bbox_text});

      way["landuse"="residential"]({bbox_text});
      relation["landuse"="residential"]({bbox_text});

      way["highway"~"^(motorway|trunk|primary|secondary|tertiary)$"]({bbox_text});
      relation["highway"~"^(motorway|trunk|primary|secondary|tertiary)$"]({bbox_text});

      node["power"="substation"]({bbox_text});
      way["power"="substation"]({bbox_text});
      relation["power"="substation"]({bbox_text});
    );
    out center tags;
    """
    body = parse.urlencode({"data": query}).encode("utf-8")
    req = request.Request(endpoint, data=body, headers={"User-Agent": "EVChargingPlanning/0.1"})
    with request.urlopen(req, timeout=timeout_seconds + 30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    payload["downloaded_at_unix"] = time.time()
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def build_barnet_real_case(
    raw_json_path: str | Path,
    bbox: dict[str, float],
    n_grid_lon: int = 24,
    n_grid_lat: int = 24,
    max_candidate_sites: int = 140,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """Build demand points, candidate sites, and existing stations from raw OSM data."""
    raw = json.loads(Path(raw_json_path).read_text(encoding="utf-8"))
    features = osm_elements_to_feature_tables(raw.get("elements", []))
    demand = _build_demand_grid(features, bbox, n_grid_lon=n_grid_lon, n_grid_lat=n_grid_lat)
    existing = _build_existing_stations(features["charging"])
    candidates = _build_candidate_sites(features, demand, max_candidate_sites=max_candidate_sites)
    return demand, candidates, existing, features


def osm_elements_to_feature_tables(elements: list[dict[str, Any]]) -> dict[str, pd.DataFrame]:
    """Convert Overpass elements to point tables by feature category."""
    buckets: dict[str, list[dict[str, Any]]] = {
        "charging": [],
        "parking": [],
        "poi": [],
        "residential": [],
        "roads": [],
        "substations": [],
    }
    for element in elements:
        lon, lat = _element_lon_lat(element)
        if lon is None or lat is None:
            continue
        tags = element.get("tags", {})
        row = {
            "osm_id": f"{element.get('type', 'element')}/{element.get('id')}",
            "lon": lon,
            "lat": lat,
            "name": tags.get("name", ""),
            "category": _category_from_tags(tags),
            "tags": json.dumps(tags, sort_keys=True),
        }
        category = row["category"]
        if category in buckets:
            buckets[category].append(row)
        elif category in {"shop", "office", "amenity"}:
            buckets["poi"].append(row)

    return {key: pd.DataFrame(rows, columns=["osm_id", "lon", "lat", "name", "category", "tags"]) for key, rows in buckets.items()}


def _element_lon_lat(element: dict[str, Any]) -> tuple[float | None, float | None]:
    if "lon" in element and "lat" in element:
        return float(element["lon"]), float(element["lat"])
    center = element.get("center")
    if center and "lon" in center and "lat" in center:
        return float(center["lon"]), float(center["lat"])
    return None, None


def _category_from_tags(tags: dict[str, Any]) -> str:
    if tags.get("amenity") == "charging_station":
        return "charging"
    if tags.get("amenity") == "parking":
        return "parking"
    if "shop" in tags:
        return "shop"
    if "office" in tags:
        return "office"
    if tags.get("landuse") == "residential":
        return "residential"
    if "highway" in tags:
        return "roads"
    if tags.get("power") == "substation":
        return "substations"
    if "amenity" in tags:
        return "amenity"
    return "other"


def _build_demand_grid(features: dict[str, pd.DataFrame], bbox: dict[str, float], n_grid_lon: int, n_grid_lat: int) -> pd.DataFrame:
    lon_values = np.linspace(bbox["west"], bbox["east"], n_grid_lon)
    lat_values = np.linspace(bbox["south"], bbox["north"], n_grid_lat)
    rows = []
    for lat in lat_values:
        for lon in lon_values:
            rows.append({"lon": lon, "lat": lat})
    demand = pd.DataFrame(rows)

    residential = _kernel_density(demand, features["residential"], radius_km=1.4)
    commercial = _kernel_density(demand, features["poi"], radius_km=1.1)
    parking = _kernel_density(demand, features["parking"], radius_km=1.0)
    roads = _inverse_distance_score(demand, features["roads"], max_distance_km=2.2)
    charging_density = _kernel_density(demand, features["charging"], radius_km=1.4)
    substation_access = _inverse_distance_score(demand, features["substations"], max_distance_km=4.0, fallback=0.45)

    residential_score = min_max_scale(residential, 0.05, 1.0)
    commercial_score = min_max_scale(commercial, 0.05, 1.0)
    road_score = min_max_scale(0.7 * roads + 0.3 * parking, 0.05, 1.0)
    accessibility = min_max_scale(0.5 * road_score + 0.35 * parking + 0.15 * commercial_score, 0.05, 1.0)
    peak_proxy = min_max_scale(0.45 * residential_score + 0.35 * commercial_score + 0.20 * road_score, 0.2, 1.0)
    hosting_capacity = min_max_scale(0.65 * substation_access + 0.35 * (1.0 - charging_density), 0.25, 1.0)
    grid_risk = min_max_scale((peak_proxy + 0.25 * charging_density) / np.maximum(hosting_capacity, 0.1), 0.0, 1.0)
    demand_value = 5.0 + 35.0 * min_max_scale(0.45 * residential_score + 0.35 * commercial_score + 0.20 * road_score)

    demand["demand_id"] = [f"ldn_d{i:04d}" for i in range(len(demand))]
    demand["residential_score"] = residential_score
    demand["commercial_score"] = commercial_score
    demand["road_score"] = road_score
    demand["accessibility"] = accessibility
    demand["hosting_capacity_proxy"] = hosting_capacity
    demand["peak_load_proxy"] = peak_proxy
    demand["grid_risk"] = grid_risk
    demand["demand"] = demand_value
    return demand


def _kernel_density(points: pd.DataFrame, features: pd.DataFrame, radius_km: float) -> np.ndarray:
    if features.empty:
        return np.zeros(len(points), dtype=float)
    distances = pairwise_haversine_km(points["lon"], points["lat"], features["lon"], features["lat"])
    weights = np.exp(-((distances / radius_km) ** 2))
    return weights.sum(axis=1)


def _inverse_distance_score(points: pd.DataFrame, features: pd.DataFrame, max_distance_km: float, fallback: float = 0.0) -> np.ndarray:
    if features.empty:
        return np.full(len(points), fallback, dtype=float)
    distances = pairwise_haversine_km(points["lon"], points["lat"], features["lon"], features["lat"])
    nearest = distances.min(axis=1)
    return np.clip(1.0 - nearest / max_distance_km, 0.0, 1.0)


def _build_existing_stations(charging: pd.DataFrame) -> pd.DataFrame:
    if charging.empty:
        return pd.DataFrame(columns=["station_id", "lon", "lat", "chargers", "source"])
    result = charging[["osm_id", "lon", "lat", "name"]].copy()
    result = result.rename(columns={"osm_id": "station_id"})
    result["chargers"] = 2
    result["source"] = "openstreetmap"
    return result


def _build_candidate_sites(features: dict[str, pd.DataFrame], demand: pd.DataFrame, max_candidate_sites: int) -> pd.DataFrame:
    candidate_frames = []
    for key in ("parking", "charging"):
        frame = features[key]
        if not frame.empty:
            candidate_frames.append(frame[["lon", "lat"]].copy())
    top_demand = demand.sort_values("demand", ascending=False).head(max(20, max_candidate_sites // 3))[["lon", "lat"]]
    candidate_frames.append(top_demand.copy())

    candidates = pd.concat(candidate_frames, ignore_index=True) if candidate_frames else top_demand.copy()
    candidates["lon_round"] = candidates["lon"].round(4)
    candidates["lat_round"] = candidates["lat"].round(4)
    candidates = candidates.drop_duplicates(["lon_round", "lat_round"]).drop(columns=["lon_round", "lat_round"])
    candidates = candidates.head(max_candidate_sites).reset_index(drop=True)

    grid_scores = _nearest_grid_values(candidates, demand)
    candidates["site_id"] = [f"ldn_c{i:04d}" for i in range(len(candidates))]
    candidates["grid_risk"] = grid_scores["grid_risk"]
    candidates["accessibility"] = grid_scores["accessibility"]
    candidates["fixed_cost"] = 62000 + 65000 * candidates["grid_risk"] + 14000 * (1.0 - candidates["accessibility"])
    candidates["charger_cost"] = 9000 + 2200 * candidates["grid_risk"]
    candidates["max_chargers"] = np.maximum(8, np.round(10 + 12 * candidates["accessibility"]).astype(int))
    return candidates[["site_id", "lon", "lat", "grid_risk", "accessibility", "fixed_cost", "charger_cost", "max_chargers"]]


def _nearest_grid_values(points: pd.DataFrame, demand: pd.DataFrame) -> pd.DataFrame:
    distances = pairwise_haversine_km(points["lon"], points["lat"], demand["lon"], demand["lat"])
    nearest = distances.argmin(axis=1)
    return demand.iloc[nearest][["grid_risk", "accessibility"]].reset_index(drop=True)
