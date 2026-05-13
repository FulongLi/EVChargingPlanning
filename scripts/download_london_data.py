#!/usr/bin/env python3
"""Download and process real London/Barnet open data."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evcs_planning.config import ensure_dirs, load_yaml
from evcs_planning.data.loaders import save_csv
from evcs_planning.data.london import boundary_json_to_lines, build_barnet_real_case, fetch_barnet_boundary_raw, fetch_barnet_osm_raw


def main() -> None:
    config = load_yaml(ROOT / "configs" / "london_barnet.yaml")
    london = config.get("london", {})
    bbox = london.get("bbox", {})
    raw_path = ROOT / london.get("raw_osm_json", "data/raw/barnet_osm_overpass.json")
    boundary_path = ROOT / london.get("boundary_osm_json", "data/raw/barnet_boundary_overpass.json")
    endpoint = london.get("overpass_endpoint", "https://overpass-api.de/api/interpreter")

    ensure_dirs([ROOT / "data" / "raw", ROOT / "data" / "processed"])
    print(f"Downloading OSM/Overpass data to {raw_path}")
    fetch_barnet_osm_raw(bbox=bbox, output_path=raw_path, endpoint=endpoint)
    print(f"Downloading Barnet boundary to {boundary_path}")
    fetch_barnet_boundary_raw(output_path=boundary_path, endpoint=endpoint)
    boundary = boundary_json_to_lines(boundary_path)

    demand, candidates, existing, features = build_barnet_real_case(
        raw_json_path=raw_path,
        bbox=bbox,
        n_grid_lon=int(london.get("n_grid_lon", 24)),
        n_grid_lat=int(london.get("n_grid_lat", 24)),
        max_candidate_sites=int(london.get("max_candidate_sites", 140)),
        boundary=boundary,
    )
    save_csv(demand, ROOT / "data" / "processed" / "london_barnet_demand_points.csv")
    save_csv(candidates, ROOT / "data" / "processed" / "london_barnet_candidate_sites.csv")
    save_csv(existing, ROOT / "data" / "processed" / "london_barnet_existing_stations.csv")
    save_csv(boundary, ROOT / "data" / "processed" / "london_barnet_boundary_lines.csv")
    for name, table in features.items():
        save_csv(table, ROOT / "data" / "processed" / f"london_barnet_osm_{name}.csv")
    print("London/Barnet real-data tables written to data/processed/")


if __name__ == "__main__":
    main()
