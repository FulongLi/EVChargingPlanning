"""London/Barnet open-data interface.

The first real-data pipeline uses OpenStreetMap data through Overpass. It keeps
the data model compatible with the synthetic pipeline so the same clustering
and optimisation modules can be reused.
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


def fetch_barnet_boundary_raw(
    output_path: str | Path,
    endpoint: str = DEFAULT_OVERPASS_ENDPOINT,
    timeout_seconds: int = 120,
) -> dict[str, Any]:
    """Download the Barnet administrative boundary from OpenStreetMap."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    query = f"""
    [out:json][timeout:{timeout_seconds}];
    relation["boundary"="administrative"]["admin_level"="8"]["name"="London Borough of Barnet"];
    out geom;
    """
    body = parse.urlencode({"data": query}).encode("utf-8")
    req = request.Request(endpoint, data=body, headers={"User-Agent": "EVChargingPlanning/0.1"})
    with request.urlopen(req, timeout=timeout_seconds + 30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    payload["downloaded_at_unix"] = time.time()
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def boundary_json_to_lines(raw_json_path: str | Path) -> pd.DataFrame:
    """Convert an Overpass relation with geometries into drawable boundary lines."""
    raw = json.loads(Path(raw_json_path).read_text(encoding="utf-8"))
    rows = []
    part_id = 0
    for element in raw.get("elements", []):
        for member in element.get("members", []):
            geometry = member.get("geometry", [])
            if not geometry:
                continue
            for point in geometry:
                rows.append({"part": part_id, "lon": float(point["lon"]), "lat": float(point["lat"])})
            part_id += 1
    return pd.DataFrame(rows, columns=["part", "lon", "lat"])


def build_barnet_real_case(
    raw_json_path: str | Path,
    bbox: dict[str, float],
    n_grid_lon: int = 24,
    n_grid_lat: int = 24,
    max_candidate_sites: int = 140,
    boundary: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """Build demand points, candidate sites, and existing stations from raw OSM data."""
    raw = json.loads(Path(raw_json_path).read_text(encoding="utf-8"))
    features = osm_elements_to_feature_tables(raw.get("elements", []))
    if boundary is not None and not boundary.empty:
        features = _filter_features_to_boundary(features, boundary)
    demand = _build_demand_grid(features, bbox, n_grid_lon=n_grid_lon, n_grid_lat=n_grid_lat)
    if boundary is not None and not boundary.empty:
        demand = clip_points_to_boundary(demand, boundary).reset_index(drop=True)
        demand["demand_id"] = [f"ldn_d{i:04d}" for i in range(len(demand))]
    existing = _build_existing_stations(features["charging"])
    candidates = _build_candidate_sites(features, demand, max_candidate_sites=max_candidate_sites)
    if boundary is not None and not boundary.empty:
        existing = clip_points_to_boundary(existing, boundary).reset_index(drop=True)
        candidates = clip_points_to_boundary(candidates, boundary).reset_index(drop=True)
        candidates["site_id"] = [f"ldn_c{i:04d}" for i in range(len(candidates))]
    return demand, candidates, existing, features


def clip_points_to_boundary(points: pd.DataFrame, boundary: pd.DataFrame) -> pd.DataFrame:
    """Keep only points inside the stitched Barnet boundary polygon."""
    if points.empty or boundary.empty:
        return points.copy()
    polygon = boundary_to_polygon(boundary)
    mask = [_point_in_polygon(float(row.lon), float(row.lat), polygon) for row in points.itertuples()]
    return points.loc[mask].copy()


def boundary_to_polygon(boundary: pd.DataFrame) -> list[tuple[float, float]]:
    """Stitch Overpass boundary line parts into a single polygon ring."""
    parts = []
    for part_id, group in boundary.groupby("part"):
        coords = [(round(float(row.lon), 7), round(float(row.lat), 7)) for row in group.itertuples()]
        if len(coords) > 1:
            parts.append((part_id, coords))
    if not parts:
        return []

    _, ring = parts.pop(0)
    while parts:
        end = ring[-1]
        match = None
        for index, (_, coords) in enumerate(parts):
            if coords[0] == end:
                match = (index, coords, False)
                break
            if coords[-1] == end:
                match = (index, coords, True)
                break
        if match is None:
            # OSM ways should join exactly after rounding. If not, join the nearest
            # endpoint to keep the clipping polygon usable.
            candidates = []
            for index, (_, coords) in enumerate(parts):
                candidates.append((_squared_distance(end, coords[0]), index, False))
                candidates.append((_squared_distance(end, coords[-1]), index, True))
            _, index, reverse = min(candidates, key=lambda item: item[0])
            coords = parts[index][1]
            match = (index, coords, reverse)
        index, coords, reverse = match
        parts.pop(index)
        if reverse:
            coords = list(reversed(coords))
        ring.extend(coords[1:] if coords[0] == ring[-1] else coords)
    if ring[0] != ring[-1]:
        ring.append(ring[0])
    return ring


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


def _filter_features_to_boundary(features: dict[str, pd.DataFrame], boundary: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {name: clip_points_to_boundary(frame, boundary) for name, frame in features.items()}


def _point_in_polygon(lon: float, lat: float, polygon: list[tuple[float, float]]) -> bool:
    if len(polygon) < 3:
        return True
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        intersects = (yi > lat) != (yj > lat) and lon < (xj - xi) * (lat - yi) / ((yj - yi) or 1e-12) + xi
        if intersects:
            inside = not inside
        j = i
    return inside


def _squared_distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return (left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2


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
            subset = frame[["lon", "lat"]].copy()
            subset["source"] = key
            candidate_frames.append(subset)
    top_demand = demand.sort_values("demand", ascending=False).head(max(20, max_candidate_sites // 3))[["lon", "lat"]]
    top_demand["source"] = "demand_hotspot"
    candidate_frames.append(top_demand.copy())

    candidates = pd.concat(candidate_frames, ignore_index=True) if candidate_frames else top_demand.copy()
    candidates["lon_round"] = candidates["lon"].round(4)
    candidates["lat_round"] = candidates["lat"].round(4)
    candidates = candidates.drop_duplicates(["lon_round", "lat_round"]).drop(columns=["lon_round", "lat_round"])

    grid_scores = _nearest_grid_values(candidates, demand)
    candidates["grid_risk"] = grid_scores["grid_risk"]
    candidates["accessibility"] = grid_scores["accessibility"]
    candidates["local_demand_score"] = _weighted_kernel_density(candidates, demand, radius_km=1.3)
    candidates["candidate_score"] = (
        0.62 * min_max_scale(candidates["local_demand_score"].to_numpy())
        + 0.28 * candidates["accessibility"].to_numpy()
        + 0.10 * (1.0 - candidates["grid_risk"].to_numpy())
    )
    candidates = _select_diverse_candidates(candidates, max_count=max_candidate_sites, min_distance_km=0.55)
    candidates["site_id"] = [f"ldn_c{i:04d}" for i in range(len(candidates))]
    candidates["fixed_cost"] = 62000 + 65000 * candidates["grid_risk"] + 14000 * (1.0 - candidates["accessibility"])
    candidates["charger_cost"] = 9000 + 2200 * candidates["grid_risk"]
    candidates["max_chargers"] = np.maximum(8, np.round(10 + 12 * candidates["accessibility"]).astype(int))
    return candidates[
        [
            "site_id",
            "lon",
            "lat",
            "source",
            "local_demand_score",
            "candidate_score",
            "grid_risk",
            "accessibility",
            "fixed_cost",
            "charger_cost",
            "max_chargers",
        ]
    ]


def _nearest_grid_values(points: pd.DataFrame, demand: pd.DataFrame) -> pd.DataFrame:
    distances = pairwise_haversine_km(points["lon"], points["lat"], demand["lon"], demand["lat"])
    nearest = distances.argmin(axis=1)
    return demand.iloc[nearest][["grid_risk", "accessibility"]].reset_index(drop=True)


def _weighted_kernel_density(points: pd.DataFrame, demand: pd.DataFrame, radius_km: float) -> np.ndarray:
    distances = pairwise_haversine_km(points["lon"], points["lat"], demand["lon"], demand["lat"])
    weights = np.exp(-((distances / radius_km) ** 2))
    return (weights * demand["demand"].to_numpy()[None, :]).sum(axis=1)


def _select_diverse_candidates(candidates: pd.DataFrame, max_count: int, min_distance_km: float) -> pd.DataFrame:
    ordered = candidates.sort_values("candidate_score", ascending=False).reset_index(drop=True)
    selected_rows = []
    selected_lon: list[float] = []
    selected_lat: list[float] = []

    for _, row in ordered.iterrows():
        if len(selected_rows) >= max_count:
            break
        if not selected_rows:
            selected_rows.append(row)
            selected_lon.append(float(row["lon"]))
            selected_lat.append(float(row["lat"]))
            continue
        distances = pairwise_haversine_km(
            [float(row["lon"])],
            [float(row["lat"])],
            selected_lon,
            selected_lat,
        )
        if float(distances.min()) >= min_distance_km:
            selected_rows.append(row)
            selected_lon.append(float(row["lon"]))
            selected_lat.append(float(row["lat"]))

    if len(selected_rows) < max_count:
        selected_ids = {row.name for row in selected_rows}
        for index, row in ordered.iterrows():
            if len(selected_rows) >= max_count:
                break
            if index not in selected_ids:
                selected_rows.append(row)

    return pd.DataFrame(selected_rows).reset_index(drop=True)
