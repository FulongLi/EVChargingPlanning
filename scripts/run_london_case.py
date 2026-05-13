#!/usr/bin/env python3
"""Run the London/Barnet real-data case study."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd

from evcs_planning.clustering.grid_aware_clustering import grid_aware_demand_clustering
from evcs_planning.clustering.kmeans import geographical_kmeans
from evcs_planning.clustering.weighted_kmeans import demand_weighted_kmeans
from evcs_planning.config import ensure_dirs, load_yaml
from evcs_planning.data.loaders import save_csv
from evcs_planning.data.london import build_barnet_real_case, fetch_barnet_osm_raw
from evcs_planning.demand.demand_model import prepare_demand_points
from evcs_planning.evaluation.metrics import comparison_table, evaluate_solution
from evcs_planning.optimisation.solver import params_from_mapping, solve_clustered_planning
from evcs_planning.visualisation.png import write_cluster_map_png, write_metric_bars_png


def main() -> None:
    config = load_yaml(ROOT / "configs" / "london_barnet.yaml")
    london = config.get("london", {})
    bbox = london.get("bbox", {})
    raw_path = ROOT / london.get("raw_osm_json", "data/raw/barnet_osm_overpass.json")
    endpoint = london.get("overpass_endpoint", "https://overpass-api.de/api/interpreter")
    output_dirs = {
        "processed": ROOT / "data" / "processed",
        "figures": ROOT / "results" / "figures",
        "tables": ROOT / "results" / "tables",
        "logs": ROOT / "results" / "logs",
    }
    ensure_dirs(output_dirs)

    if not raw_path.exists():
        print(f"Raw OSM file not found; downloading {raw_path}")
        fetch_barnet_osm_raw(bbox=bbox, output_path=raw_path, endpoint=endpoint)

    demand_raw, candidates, existing, features = build_barnet_real_case(
        raw_json_path=raw_path,
        bbox=bbox,
        n_grid_lon=int(london.get("n_grid_lon", 24)),
        n_grid_lat=int(london.get("n_grid_lat", 24)),
        max_candidate_sites=int(london.get("max_candidate_sites", 140)),
    )
    demand = prepare_demand_points(demand_raw)
    save_csv(demand, output_dirs["processed"] / "london_barnet_demand_points.csv")
    save_csv(candidates, output_dirs["processed"] / "london_barnet_candidate_sites.csv")
    save_csv(existing, output_dirs["processed"] / "london_barnet_existing_stations.csv")
    for name, table in features.items():
        save_csv(table, output_dirs["processed"] / f"london_barnet_osm_{name}.csv")

    random_seed = int(config.get("random_seed", 42))
    clustering_config = config.get("clustering", {})
    n_clusters = int(clustering_config.get("n_clusters", 7))
    max_iter = int(clustering_config.get("max_iter", 100))
    params = params_from_mapping(config)
    methods = {
        "london_geographical_kmeans": lambda: geographical_kmeans(demand, n_clusters, random_seed, max_iter),
        "london_demand_weighted_kmeans": lambda: demand_weighted_kmeans(demand, n_clusters, random_seed, max_iter),
        "london_grid_aware_demand_clustering": lambda: grid_aware_demand_clustering(
            demand,
            n_clusters,
            random_seed,
            max_iter,
            feature_weights=clustering_config.get("feature_weights", {}),
        ),
    }

    metrics = {}
    selected_tables = []
    assignment_tables = []
    for method_name, runner in methods.items():
        clustering = runner()
        selected, assignments = solve_clustered_planning(demand, candidates, clustering.labels, params, method_name)
        metrics[method_name] = evaluate_solution(selected, assignments, params)
        selected_tables.append(selected)
        assignment_tables.append(assignments)
        labelled = demand.copy()
        labelled["cluster"] = clustering.labels
        save_csv(labelled, output_dirs["processed"] / f"{method_name}_demand_clusters.csv")
        write_cluster_map_png(
            demand,
            clustering.labels,
            candidates,
            selected,
            output_dirs["figures"] / f"{method_name}_map.png",
        )

    comparison = comparison_table(metrics)
    save_csv(comparison, output_dirs["tables"] / "london_barnet_summary_metrics.csv")
    save_csv(pd.concat(selected_tables, ignore_index=True), output_dirs["tables"] / "london_barnet_selected_sites.csv")
    save_csv(pd.concat(assignment_tables, ignore_index=True), output_dirs["tables"] / "london_barnet_assignments.csv")
    write_metric_bars_png(
        comparison,
        "average_service_distance_km",
        output_dirs["figures"] / "london_average_service_distance_comparison.png",
    )
    write_metric_bars_png(
        comparison,
        "grid_impact_per_charger",
        output_dirs["figures"] / "london_grid_impact_per_charger_comparison.png",
    )
    print(f"London/Barnet case complete: {output_dirs['tables'] / 'london_barnet_summary_metrics.csv'}")


if __name__ == "__main__":
    main()
