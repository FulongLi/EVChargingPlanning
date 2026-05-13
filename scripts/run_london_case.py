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
from evcs_planning.data.london import boundary_json_to_lines, build_barnet_real_case, fetch_barnet_boundary_raw, fetch_barnet_osm_raw
from evcs_planning.demand.demand_model import prepare_demand_points
from evcs_planning.evaluation.metrics import comparison_table, evaluate_solution
from evcs_planning.optimisation.solver import params_from_mapping, solve_clustered_planning, solve_preselected_sites
from evcs_planning.visualisation.png import write_cluster_map_png, write_metric_bars_png, write_method_flowchart_png, write_tradeoff_scatter_png


def main() -> None:
    config = load_yaml(ROOT / "configs" / "london_barnet.yaml")
    london = config.get("london", {})
    bbox = london.get("bbox", {})
    raw_path = ROOT / london.get("raw_osm_json", "data/raw/barnet_osm_overpass.json")
    boundary_raw_path = ROOT / london.get("boundary_osm_json", "data/raw/barnet_boundary_overpass.json")
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
    if not boundary_raw_path.exists():
        print(f"Boundary OSM file not found; downloading {boundary_raw_path}")
        fetch_barnet_boundary_raw(output_path=boundary_raw_path, endpoint=endpoint)
    boundary = boundary_json_to_lines(boundary_raw_path)

    demand_raw, candidates, existing, features = build_barnet_real_case(
        raw_json_path=raw_path,
        bbox=bbox,
        n_grid_lon=int(london.get("n_grid_lon", 24)),
        n_grid_lat=int(london.get("n_grid_lat", 24)),
        max_candidate_sites=int(london.get("max_candidate_sites", 140)),
        boundary=boundary,
    )
    demand = prepare_demand_points(demand_raw)
    save_csv(demand, output_dirs["processed"] / "london_barnet_demand_points.csv")
    save_csv(candidates, output_dirs["processed"] / "london_barnet_candidate_sites.csv")
    save_csv(existing, output_dirs["processed"] / "london_barnet_existing_stations.csv")
    save_csv(boundary, output_dirs["processed"] / "london_barnet_boundary_lines.csv")
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
            title=_method_title(method_name),
            subtitle=_method_subtitle(method_name),
            boundary=boundary,
        )

    baseline_count = int(pd.concat(selected_tables, ignore_index=True).query("method == 'london_grid_aware_demand_clustering'").shape[0])
    baselines = _build_direct_baselines(candidates, baseline_count, random_seed)
    baseline_labels = [0] * len(demand)
    for method_name, selected_candidates in baselines.items():
        selected, assignments = solve_preselected_sites(demand, selected_candidates, params, method_name)
        metrics[method_name] = evaluate_solution(selected, assignments, params)
        selected_tables.append(selected)
        assignment_tables.append(assignments)
        write_cluster_map_png(
            demand,
            baseline_labels,
            candidates,
            selected,
            output_dirs["figures"] / f"{method_name}_map.png",
            title=_method_title(method_name),
            subtitle=_method_subtitle(method_name),
            boundary=boundary,
        )

    comparison = comparison_table(metrics)
    save_csv(comparison, output_dirs["tables"] / "london_barnet_summary_metrics.csv")
    save_csv(pd.concat(selected_tables, ignore_index=True), output_dirs["tables"] / "london_barnet_selected_sites.csv")
    save_csv(pd.concat(assignment_tables, ignore_index=True), output_dirs["tables"] / "london_barnet_assignments.csv")
    write_metric_bars_png(
        comparison,
        "average_service_distance_km",
        output_dirs["figures"] / "london_average_service_distance_comparison.png",
        title="London/Barnet Average Service Distance",
        note="Based on demand-weighted distance from each demand grid point to its assigned selected station.",
    )
    write_metric_bars_png(
        comparison,
        "grid_impact_per_charger",
        output_dirs["figures"] / "london_grid_impact_per_charger_comparison.png",
        title="London/Barnet Grid Impact Per Charger",
        note="Based on selected station grid-risk proxy multiplied by installed charging power.",
    )
    write_tradeoff_scatter_png(
        comparison,
        x_metric="average_service_distance_km",
        y_metric="grid_impact_per_charger",
        path=output_dirs["figures"] / "london_distance_grid_tradeoff.png",
        title="London/Barnet Distance-Grid Tradeoff",
        note="Each point is one method. Lower-left means shorter distance and lower grid risk.",
    )
    write_method_flowchart_png(output_dirs["figures"] / "gdc_lco_method_flowchart.png")
    print(f"London/Barnet case complete: {output_dirs['tables'] / 'london_barnet_summary_metrics.csv'}")


def _build_direct_baselines(candidates: pd.DataFrame, count: int, random_seed: int) -> dict[str, pd.DataFrame]:
    count = max(1, min(count, len(candidates)))
    hotspot = candidates.sort_values("local_demand_score", ascending=False).head(count)
    existing = candidates[candidates.get("source", "") == "charging"].sort_values("local_demand_score", ascending=False).head(count)
    if len(existing) < count:
        existing = pd.concat([existing, candidates.sort_values("local_demand_score", ascending=False)]).drop_duplicates("site_id").head(count)
    random_sites = candidates.sample(n=count, random_state=random_seed)
    return {
        "london_demand_hotspot_baseline": hotspot,
        "london_existing_expansion_baseline": existing,
        "london_random_candidate_baseline": random_sites,
    }


def _method_title(method_name: str) -> str:
    titles = {
        "london_geographical_kmeans": "BASIS GEOGRAPHICAL K-MEANS",
        "london_demand_weighted_kmeans": "BASIS DEMAND-WEIGHTED K-MEANS",
        "london_grid_aware_demand_clustering": "BASIS GRID-AWARE CLUSTERING",
        "london_demand_hotspot_baseline": "BASIS DEMAND-HOTSPOT BASELINE",
        "london_existing_expansion_baseline": "BASIS EXISTING-STATION BASELINE",
        "london_random_candidate_baseline": "BASIS RANDOM BASELINE",
    }
    return titles.get(method_name, method_name.replace("_", " ").title())


def _method_subtitle(method_name: str) -> str:
    subtitles = {
        "london_geographical_kmeans": "INPUTS LON AND LAT ONLY THEN SITE SELECTION BY ZONE",
        "london_demand_weighted_kmeans": "INPUTS LON LAT AND DEMAND WEIGHTS HIGH DEMAND PULLS ZONES",
        "london_grid_aware_demand_clustering": "INPUTS OSM DEMAND ACCESSIBILITY AND GRID-RISK PROXY",
        "london_demand_hotspot_baseline": "INPUTS LOCAL DEMAND SCORE ONLY NO CLUSTERING",
        "london_existing_expansion_baseline": "INPUTS EXISTING OSM CHARGING LOCATIONS FIRST",
        "london_random_candidate_baseline": "INPUTS RANDOM CANDIDATE SITES LOWER-BOUND BASELINE",
    }
    return subtitles.get(method_name, "Generated from the London/Barnet open-data planning pipeline.")


if __name__ == "__main__":
    main()
