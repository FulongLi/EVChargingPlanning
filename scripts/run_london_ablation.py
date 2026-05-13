#!/usr/bin/env python3
"""Run feature ablation experiments for the London/Barnet GDC-LCO pipeline."""

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
from evcs_planning.data.london import boundary_json_to_lines, build_barnet_real_case
from evcs_planning.demand.demand_model import prepare_demand_points
from evcs_planning.evaluation.metrics import comparison_table, evaluate_solution
from evcs_planning.optimisation.solver import params_from_mapping, solve_clustered_planning
from evcs_planning.visualisation.png import write_metric_bars_png, write_tradeoff_scatter_png


def main() -> None:
    config = load_yaml(ROOT / "configs" / "london_barnet.yaml")
    london = config.get("london", {})
    raw_path = ROOT / london.get("raw_osm_json", "data/raw/barnet_osm_overpass.json")
    boundary_path = ROOT / london.get("boundary_osm_json", "data/raw/barnet_boundary_overpass.json")
    if not raw_path.exists():
        raise FileNotFoundError("Run scripts/run_london_case.py once to download the raw Overpass file.")
    if not boundary_path.exists():
        raise FileNotFoundError("Run scripts/run_london_case.py once to download the Barnet boundary file.")
    boundary = boundary_json_to_lines(boundary_path)

    output_dirs = {
        "figures": ROOT / "results" / "figures",
        "tables": ROOT / "results" / "tables",
    }
    ensure_dirs(output_dirs)
    demand_raw, candidates, _, _ = build_barnet_real_case(
        raw_json_path=raw_path,
        bbox=london.get("bbox", {}),
        n_grid_lon=int(london.get("n_grid_lon", 24)),
        n_grid_lat=int(london.get("n_grid_lat", 24)),
        max_candidate_sites=int(london.get("max_candidate_sites", 140)),
        boundary=boundary,
    )
    demand = prepare_demand_points(demand_raw)
    params = params_from_mapping(config)
    random_seed = int(config.get("random_seed", 42))
    n_clusters = int(config.get("clustering", {}).get("n_clusters", 9))
    max_iter = int(config.get("clustering", {}).get("max_iter", 100))
    grid_weight = float(config.get("clustering", {}).get("feature_weights", {}).get("grid_risk", 0.5))

    ablations = {
        "ablation_spatial_only": lambda: geographical_kmeans(demand, n_clusters, random_seed, max_iter),
        "ablation_spatial_demand_weighted": lambda: demand_weighted_kmeans(demand, n_clusters, random_seed, max_iter),
        "ablation_spatial_demand_feature": lambda: grid_aware_demand_clustering(
            demand,
            n_clusters,
            random_seed,
            max_iter,
            feature_weights={"spatial": 1.0, "demand": 0.65, "accessibility": 0.0, "grid_risk": 0.0},
        ),
        "ablation_plus_accessibility": lambda: grid_aware_demand_clustering(
            demand,
            n_clusters,
            random_seed,
            max_iter,
            feature_weights={"spatial": 1.0, "demand": 0.65, "accessibility": 0.4, "grid_risk": 0.0},
        ),
        "ablation_plus_grid_risk": lambda: grid_aware_demand_clustering(
            demand,
            n_clusters,
            random_seed,
            max_iter,
            feature_weights={"spatial": 1.0, "demand": 0.65, "accessibility": 0.4, "grid_risk": grid_weight},
        ),
    }

    results = {}
    for method_name, runner in ablations.items():
        clustering = runner()
        selected, assignments = solve_clustered_planning(demand, candidates, clustering.labels, params, method_name)
        metrics = evaluate_solution(selected, assignments, params)
        metrics["clustering_inertia"] = clustering.inertia
        results[method_name] = metrics

    table = comparison_table(results)
    save_csv(table, output_dirs["tables"] / "london_feature_ablation.csv")
    write_metric_bars_png(
        table,
        "grid_impact_per_charger",
        output_dirs["figures"] / "london_ablation_grid_impact_per_charger.png",
        title="London Ablation: Grid Impact Per Charger",
        note="Adds feature groups step by step. Lower bars indicate lower grid-risk exposure per charger.",
    )
    write_tradeoff_scatter_png(
        table,
        "average_service_distance_km",
        "grid_impact_per_charger",
        output_dirs["figures"] / "london_ablation_distance_grid_tradeoff.png",
        title="London Ablation Distance-Grid Tradeoff",
        note="Each point adds feature groups. Lower-left is better.",
    )
    print(f"Ablation analysis complete: {output_dirs['tables'] / 'london_feature_ablation.csv'}")


if __name__ == "__main__":
    main()
