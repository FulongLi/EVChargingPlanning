# Grid-Aware EV Charging Station Planning

This repository implements a reproducible research pipeline for electric vehicle charging station siting and sizing using grid-aware demand clustering.

The working paper title is:

> Grid-Aware Demand Clustering for Electric Vehicle Charging Station Siting and Sizing: A London Case Study

The Chinese title is:

> 基于电网感知需求聚类的电动汽车充电站选址定容方法：伦敦案例研究

## Research Idea

Urban EV charging demand is spatially uneven and is shaped by residential density, commercial activity, road accessibility, parking availability, and existing charging infrastructure. Conventional siting models often focus on distance or coverage while treating grid-side feasibility as a later constraint.

This project uses clustering as a problem decomposition tool. Demand points are first partitioned into interpretable charging service zones using demand, accessibility, and grid-risk proxy features. Location-capacity optimisation is then solved within each zone.

## Methodology

```text
Urban data
  -> spatial preprocessing
  -> demand feature construction
  -> geographical / demand-weighted / grid-aware clustering
  -> candidate charging station generation
  -> location-capacity optimisation
  -> evaluation and visualisation
```

The proposed method is named GDC-LCO:

> Grid-aware Demand Clustering for Location-Capacity Optimisation

## Current MVP

The first version runs fully on synthetic data so the complete pipeline can be tested before London open data is integrated.

It includes:

- synthetic demand, candidate site, existing station, accessibility, and grid-risk data
- geographical K-means
- demand-weighted K-means
- grid-aware demand clustering
- simplified greedy location-capacity optimisation
- baseline comparison metrics
- SVG figures and CSV result tables
- LaTeX paper skeleton

## Install

```bash
python3 -m pip install -e .
```

If you do not want to install the package yet, the scripts also add `src/` to `PYTHONPATH` automatically.

## Run Synthetic Case

```bash
python3 scripts/run_synthetic_case.py
```

Outputs are written to:

```text
data/processed/
results/figures/
results/tables/
results/logs/
```

## Run London Case

The London interface is intentionally a placeholder in the MVP. The core algorithms do not hard-code London-specific assumptions; real London data should be connected through configuration and data loader modules.

```bash
python3 scripts/run_london_case.py
```

## Repository Structure

```text
src/evcs_planning/
  data/            data loading, preprocessing, synthetic generation
  demand/          demand feature construction
  clustering/      K-means, demand-weighted K-means, grid-aware clustering
  optimisation/    candidate generation and location-capacity solver
  evaluation/      metrics and baseline helpers
  visualisation/   SVG maps, charts, and result tables

scripts/           runnable experiment scripts
configs/           case-study and model parameters
docs/              methodology and experiment notes
paper/             LaTeX paper skeleton
```

## Citation

Citation information will be added when the manuscript draft is ready.
