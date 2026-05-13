#!/usr/bin/env python3
"""Run the synthetic GDC-LCO case."""

from __future__ import annotations

import json
import sys
import time
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
from evcs_planning.data.synthetic import config_from_mapping, generate_synthetic_case
from evcs_planning.demand.demand_model import prepare_demand_points
from evcs_planning.evaluation.metrics import comparison_table, evaluate_solution
from evcs_planning.optimisation.solver import params_from_mapping, solve_clustered_planning
from evcs_planning.visualisation.png import write_cluster_map_png, write_metric_bars_png


def main() -> None:
    config_path = ROOT / "configs" / "synthetic.yaml"
    config = load_yaml(config_path)
    random_seed = int(config.get("random_seed", 42))
    n_clusters = int(config.get("clustering", {}).get("n_clusters", 7))
    max_iter = int(config.get("clustering", {}).get("max_iter", 100))
    feature_weights = config.get("clustering", {}).get("feature_weights", {})
    params = params_from_mapping(config)

    output_dirs = {
        "processed": ROOT / "data" / "processed",
        "figures": ROOT / "results" / "figures",
        "tables": ROOT / "results" / "tables",
        "logs": ROOT / "results" / "logs",
    }
    ensure_dirs(output_dirs)

    demand_raw, candidates, existing = generate_synthetic_case(config_from_mapping(config))
    demand = prepare_demand_points(demand_raw)

    save_csv(demand, output_dirs["processed"] / "synthetic_demand_points.csv")
    save_csv(candidates, output_dirs["processed"] / "synthetic_candidate_sites.csv")
    save_csv(existing, output_dirs["processed"] / "synthetic_existing_stations.csv")

    methods = {
        "geographical_kmeans": lambda: geographical_kmeans(demand, n_clusters, random_seed, max_iter),
        "demand_weighted_kmeans": lambda: demand_weighted_kmeans(demand, n_clusters, random_seed, max_iter),
        "grid_aware_demand_clustering": lambda: grid_aware_demand_clustering(
            demand,
            n_clusters,
            random_seed,
            max_iter,
            feature_weights=feature_weights,
        ),
    }

    metrics: dict[str, dict[str, float]] = {}
    selected_tables = []
    assignment_tables = []
    log_rows = []

    for method_name, runner in methods.items():
        started = time.perf_counter()
        clustering = runner()
        selected, assignments = solve_clustered_planning(demand, candidates, clustering.labels, params, method_name)
        elapsed = time.perf_counter() - started
        method_metrics = evaluate_solution(selected, assignments, params)
        method_metrics["computation_time_seconds"] = elapsed
        method_metrics["clustering_inertia"] = clustering.inertia
        metrics[method_name] = method_metrics

        selected_tables.append(selected)
        assignment_tables.append(assignments)
        log_rows.append(
            {
                "method": method_name,
                "elapsed_seconds": elapsed,
                "inertia": clustering.inertia,
                "feature_names": ",".join(clustering.feature_names),
            }
        )

        labelled_demand = demand.copy()
        labelled_demand["cluster"] = clustering.labels
        save_csv(labelled_demand, output_dirs["processed"] / f"{method_name}_demand_clusters.csv")
        write_cluster_map_png(
            demand,
            clustering.labels,
            candidates,
            selected,
            output_dirs["figures"] / f"{method_name}_map.png",
            title=f"BASIS {method_name.replace('_', ' ').upper()}",
            subtitle="INPUTS SYNTHETIC DEMAND CANDIDATES ACCESSIBILITY AND GRID RISK",
            note="DOTS DEMAND ZONES  RINGS CANDIDATES  BLACK SELECTED STATIONS",
        )

    comparison = comparison_table(metrics)
    selected_all = pd.concat(selected_tables, ignore_index=True)
    assignments_all = pd.concat(assignment_tables, ignore_index=True)

    save_csv(comparison, output_dirs["tables"] / "synthetic_summary_metrics.csv")
    save_csv(selected_all, output_dirs["tables"] / "synthetic_selected_sites.csv")
    save_csv(assignments_all, output_dirs["tables"] / "synthetic_assignments.csv")
    save_csv(pd.DataFrame(log_rows), output_dirs["logs"] / "synthetic_run_log.csv")

    write_metric_bars_png(
        comparison,
        "average_service_distance_km",
        output_dirs["figures"] / "average_service_distance_comparison.png",
        title="Synthetic Case Average Service Distance",
        note="Based on demand-weighted distance from each synthetic demand point to its assigned station.",
    )
    write_metric_bars_png(
        comparison,
        "grid_impact_score",
        output_dirs["figures"] / "grid_impact_comparison.png",
        title="Synthetic Case Total Grid Impact",
        note="Based on installed charging power weighted by synthetic grid-risk proxy.",
    )

    print(json.dumps({"case": config.get("case_name"), "methods": list(methods), "summary": str(output_dirs["tables"] / "synthetic_summary_metrics.csv")}, indent=2))


if __name__ == "__main__":
    main()
