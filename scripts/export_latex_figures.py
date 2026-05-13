#!/usr/bin/env python3
"""Export paper figures as LaTeX/PGFPlots so figure text matches the manuscript."""

from __future__ import annotations

from pathlib import Path
import shutil
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evcs_planning.data.london import boundary_to_polygon

FIG = ROOT / "paper" / "figures"
DATA = FIG / "data"

METHODS = [
    ("london_geographical_kmeans", "Geographical K-means"),
    ("london_demand_weighted_kmeans", "Demand-weighted K-means"),
    ("london_grid_aware_demand_clustering", "Grid-aware demand clustering"),
]

COLORS = ["blue", "red", "green!60!black", "purple", "orange", "cyan!70!black", "magenta", "teal", "brown", "gray"]
LON_MIN = -0.335
LON_MAX = -0.125
LAT_MIN = 51.555
LAT_MAX = 51.690


def main() -> None:
    if DATA.exists():
        shutil.rmtree(DATA)
    DATA.mkdir(parents=True, exist_ok=True)
    for stale in FIG.glob("fig_*.tex"):
        stale.unlink()
    _export_boundary()
    candidates = pd.read_csv(ROOT / "data" / "processed" / "london_barnet_candidate_sites.csv")
    candidates[["lon", "lat"]].to_csv(DATA / "candidate_sites.tsv", sep="\t", index=False)

    selected_all = pd.read_csv(ROOT / "results" / "tables" / "london_barnet_selected_sites.csv")
    for method, title in METHODS:
        demand = pd.read_csv(ROOT / "data" / "processed" / f"{method}_demand_clusters.csv")
        for cluster, cluster_frame in demand.groupby("cluster"):
            cluster_frame[["lon", "lat"]].to_csv(DATA / f"{method}_cluster_{int(cluster)}.tsv", sep="\t", index=False)
        selected = selected_all[selected_all["method"] == method]
        selected[["lon", "lat"]].to_csv(DATA / f"{method}_selected.tsv", sep="\t", index=False)
        _write_map_tex(method, title, int(demand["cluster"].max()) + 1)

    _write_tradeoff_tex()
    _write_ablation_tex()
    _write_sensitivity_tex()
    _write_flowchart_tex()


def _export_boundary() -> None:
    boundary = pd.read_csv(ROOT / "data" / "processed" / "london_barnet_boundary_lines.csv")
    with (DATA / "barnet_boundary.tsv").open("w", encoding="utf-8") as handle:
        handle.write("lon\tlat\n")
        for lon, lat in boundary_to_polygon(boundary):
            handle.write(f"{lon}\t{lat}\n")


def _write_map_tex(method: str, title: str, n_clusters: int) -> None:
    lines = [
        r"\begin{tikzpicture}",
        r"\begin{axis}[",
        r"width=0.94\linewidth,",
        r"height=0.72\linewidth,",
        r"xlabel={Longitude},",
        r"ylabel={Latitude},",
        rf"xmin={LON_MIN}, xmax={LON_MAX},",
        rf"ymin={LAT_MIN}, ymax={LAT_MAX},",
        r"scaled ticks=false,",
        r"tick label style={/pgf/number format/fixed, font=\scriptsize},",
        r"grid=both,",
        r"major grid style={line width=.1pt,draw=gray!25},",
        r"label style={font=\scriptsize},",
        r"legend style={font=\scriptsize, at={(0.02,0.02)}, anchor=south west, fill=white, fill opacity=0.85, draw=none},",
        r"]",
        rf"\addplot[black, very thick] table[x=lon,y=lat] {{figures/data/barnet_boundary.tsv}};",
        rf"\addlegendentry{{Barnet boundary}}",
    ]
    for cluster in range(n_clusters):
        color = COLORS[cluster % len(COLORS)]
        lines.append(
            rf"\addplot[only marks, mark=*, mark size=0.75pt, {color}, opacity=0.55] "
            rf"table[x=lon,y=lat] {{figures/data/{method}_cluster_{cluster}.tsv}};"
        )
    lines.extend(
        [
            rf"\addplot[only marks, mark=o, mark size=1.1pt, draw=gray!70, opacity=0.70] "
            rf"table[x=lon,y=lat] {{figures/data/candidate_sites.tsv}};",
            rf"\addlegendentry{{Candidate sites}}",
            rf"\addplot[only marks, mark=*, mark size=2.6pt, black] "
            rf"table[x=lon,y=lat] {{figures/data/{method}_selected.tsv}};",
            rf"\addlegendentry{{Selected stations}}",
            r"\end{axis}",
            r"\end{tikzpicture}",
        ]
    )
    (FIG / f"fig_{method}_map.tex").write_text("\n".join(lines), encoding="utf-8")


def _write_tradeoff_tex() -> None:
    table = pd.read_csv(ROOT / "results" / "tables" / "london_barnet_summary_metrics.csv")
    labels = {
        "london_geographical_kmeans": "Geo. K-means",
        "london_demand_weighted_kmeans": "Demand weighted",
        "london_grid_aware_demand_clustering": "Grid-aware",
        "london_demand_hotspot_baseline": "Hotspot",
        "london_existing_expansion_baseline": "Existing",
        "london_random_candidate_baseline": "Random",
    }
    lines = _scatter_header("Average service distance (km)", "Grid impact per charger")
    for index, row in table.iterrows():
        color = COLORS[index % len(COLORS)]
        label = labels.get(row["method"], row["method"])
        x = row["average_service_distance_km"]
        y = row["grid_impact_per_charger"]
        lines.append(rf"\addplot[only marks, mark=*, mark size=2.4pt, {color}] coordinates {{({x},{y})}};")
        lines.append(rf"\addlegendentry{{{label}}}")
    lines.extend([r"\end{axis}", r"\end{tikzpicture}"])
    (FIG / "fig_london_distance_grid_tradeoff.tex").write_text("\n".join(lines), encoding="utf-8")


def _write_ablation_tex() -> None:
    table = pd.read_csv(ROOT / "results" / "tables" / "london_feature_ablation.csv")
    labels = {
        "ablation_spatial_only": "Spatial only",
        "ablation_spatial_demand_weighted": "Demand weighted",
        "ablation_spatial_demand_feature": "Demand feature",
        "ablation_plus_accessibility": "Plus accessibility",
        "ablation_plus_grid_risk": "Plus grid risk",
    }
    lines = _scatter_header("Average service distance (km)", "Grid impact per charger")
    for index, row in table.iterrows():
        color = COLORS[index % len(COLORS)]
        label = labels.get(row["method"], row["method"])
        x = row["average_service_distance_km"]
        y = row["grid_impact_per_charger"]
        lines.append(rf"\addplot[only marks, mark=*, mark size=2.4pt, {color}] coordinates {{({x},{y})}};")
        lines.append(rf"\addlegendentry{{{label}}}")
    lines.extend([r"\end{axis}", r"\end{tikzpicture}"])
    (FIG / "fig_london_ablation_tradeoff.tex").write_text("\n".join(lines), encoding="utf-8")


def _scatter_header(xlabel: str, ylabel: str) -> list[str]:
    return [
        r"\begin{tikzpicture}",
        r"\begin{axis}[",
        r"width=0.94\linewidth,",
        r"height=0.70\linewidth,",
        rf"xlabel={{{xlabel}}},",
        rf"ylabel={{{ylabel}}},",
        r"grid=both,",
        r"tick label style={font=\scriptsize},",
        r"label style={font=\scriptsize},",
        r"legend columns=2,",
        r"legend style={font=\tiny, at={(0.03,0.97)}, anchor=north west, fill=white, fill opacity=0.88, draw=none},",
        r"]",
    ]


def _write_sensitivity_tex() -> None:
    table = pd.read_csv(ROOT / "results" / "tables" / "london_grid_weight_sensitivity.csv")
    table[["grid_weight", "n_clusters", "composite_score"]].to_csv(DATA / "sensitivity.tsv", sep="\t", index=False)
    content = r"""\begin{tikzpicture}
\begin{axis}[
width=0.82\linewidth,
height=0.62\linewidth,
xlabel={Grid-risk feature weight},
ylabel={Number of clusters},
colorbar,
colormap/viridis,
point meta min=0,
point meta max=1,
tick label style={font=\scriptsize},
label style={font=\scriptsize},
]
\addplot[
matrix plot*,
mesh/cols=7,
point meta=explicit,
] table[x=grid_weight,y=n_clusters,meta=composite_score] {figures/data/sensitivity.tsv};
\end{axis}
\end{tikzpicture}
"""
    (FIG / "fig_london_sensitivity.tex").write_text(content, encoding="utf-8")


def _write_flowchart_tex() -> None:
    content = r"""\begin{tikzpicture}[
node distance=6mm,
box/.style={draw, rounded corners, align=center, minimum width=21mm, minimum height=11mm, font=\scriptsize},
arrow/.style={-Latex, thick}
]
\node[box] (data) {Open urban\\data};
\node[box, right=of data] (features) {Demand, access,\\grid-risk features};
\node[box, right=of features] (cluster) {Grid-aware\\demand clustering};
\node[box, right=of cluster] (candidate) {Candidate site\\generation};
\node[box, right=of candidate] (optimise) {Location-capacity\\optimisation};
\node[box, right=of optimise] (eval) {Evaluation and\\visualisation};
\draw[arrow] (data) -- (features);
\draw[arrow] (features) -- (cluster);
\draw[arrow] (cluster) -- (candidate);
\draw[arrow] (candidate) -- (optimise);
\draw[arrow] (optimise) -- (eval);
\end{tikzpicture}
"""
    (FIG / "fig_gdc_lco_flowchart.tex").write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
