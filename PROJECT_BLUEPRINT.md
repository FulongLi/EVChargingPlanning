# PROJECT BLUEPRINT
# Grid-Aware Clustering-Based EV Charging Station Planning

## 1. Project Goal

This project aims to develop a reproducible research package for **electric vehicle charging station siting and sizing** using a **clustering-based partitioning method**.

The research topic is:

> **Grid-Aware Demand Clustering for Electric Vehicle Charging Station Siting and Sizing: A London Case Study**

Chinese title:

> **基于电网感知需求聚类的电动汽车充电站选址定容方法：伦敦案例研究**

The goal is to build both:

1. A research paper draft.
2. An open-source Python project package that can be uploaded to GitHub.

The project should be designed as a clean, reproducible research repository, not just a collection of notebooks.

---

## 2. Research Motivation

Urban EV charging demand is spatially uneven. Demand is affected by population density, commuting behaviour, commercial activity, parking availability, existing charging infrastructure, and road accessibility.

Traditional EV charging station siting methods often focus on geographical distance or demand coverage, but they may ignore grid-side constraints such as local connection cost, hosting capacity, peak load pressure, or transformer/loading risk.

The proposed project uses clustering not as the final solution, but as a **problem decomposition tool**. The city is first divided into interpretable charging service zones, and then charging station location and capacity are optimised within each zone.

Core research idea:

> Use clustering-based partitioning to reduce the complexity of large-scale EV charging station planning while incorporating demand, accessibility, and grid-proxy information.

---

## 3. Recommended Case Study Area

After comparing London, Shenzhen, Cardiff, and Shanghai, the recommended first case study is:

> **London, especially Barnet or selected Inner London boroughs.**

Reasoning:

1. London has strong open data availability.
2. UK-based case study is easier to justify and reproduce.
3. Existing EV charging infrastructure data is easier to access.
4. Barnet may have useful EV charging transaction data.
5. London provides a good balance between data availability, complexity, and academic relevance.

Suggested implementation strategy:

```text
Stage 1: Use London Barnet as the MVP case study.
Stage 2: Extend to selected Inner London boroughs.
Stage 3: Optionally compare with Shenzhen UrbanEV data in future work.
```

Shanghai is not recommended for the first version because data acquisition and cleaning may be more difficult. Cardiff is easier but may be too small to demonstrate the value of clustering-based planning.

---

## 4. Core Research Question

The project should answer:

> How can a grid-aware clustering-based framework improve EV charging station siting and sizing by jointly considering user convenience, demand coverage, investment cost, and grid impact?

More specifically:

```text
RQ1: Can clustering-based partitioning reduce the complexity of urban EV charging station planning?
RQ2: Does demand-weighted clustering improve charging demand coverage compared with geographical clustering?
RQ3: Does adding grid-aware features reduce grid impact or connection risk?
RQ4: Can the proposed method provide better siting and sizing results than baseline strategies?
```

---

## 5. Proposed Method Name

The proposed algorithm/framework can be named:

```text
GDC-LCO
```

Full name:

> **Grid-aware Demand Clustering for Location-Capacity Optimisation**

Chinese:

> **电网感知需求聚类选址定容优化方法**

---

## 6. Overall Methodology

The full pipeline should follow this structure:

```text
Urban Open Data
      ↓
Data Cleaning and Spatial Preprocessing
      ↓
Charging Demand Feature Construction
      ↓
Grid-Aware Demand Clustering
      ↓
Candidate Charging Station Generation
      ↓
Location-Capacity Optimisation
      ↓
Evaluation and Visualisation
      ↓
Paper Figures and Tables
```

---

## 7. Data Layer

The project should use open or reproducible urban datasets.

Possible data categories:

```text
1. Existing EV charging station locations
2. EV charging transaction data if available
3. Population density
4. POI data
5. Road network data
6. Borough boundary
7. Parking or residential density data
8. Simplified grid proxy data
```

For the first version, real distribution network topology is not required. Grid information can be approximated using proxy variables.

Possible grid proxies:

```text
1. Distance to nearest assumed grid connection point
2. Local peak demand proxy
3. Existing charging station density
4. Local population and activity density
5. Assumed hosting capacity score
6. Grid connection cost proxy
```

Example proxy formulation:

```text
grid_impact_score = local_peak_demand / assumed_hosting_capacity
```

or:

```text
grid_connection_cost = distance_to_nearest_grid_node × unit_connection_cost
```

---

## 8. Demand Modelling

The project should construct charging demand points with demand weights.

Demand can be estimated from three components:

### 8.1 Residential Demand

Represents home/neighbourhood charging demand.

Possible features:

```text
population density
residential land use
car ownership proxy
households without private parking
EV adoption ratio
```

### 8.2 Commercial / Workplace Demand

Represents daytime destination charging demand.

Possible features:

```text
POI density
office density
retail density
parking facilities
employment/activity proxy
```

### 8.3 Road / Transit Demand

Represents en-route or fast-charging demand.

Possible features:

```text
road centrality
distance to major roads
traffic flow proxy
transport hubs
```

Total demand score:

```text
D_i = alpha * D_residential_i
    + beta  * D_commercial_i
    + gamma * D_road_i
```

The first version can use a rule-based demand model rather than deep learning.

---

## 9. Clustering Layer

The project should implement and compare at least three clustering methods.

### Method 1: Geographical K-means

Uses only spatial coordinates:

```text
x_i = [lon_i, lat_i]
```

This is the simplest baseline.

### Method 2: Demand-Weighted K-means

Uses demand weights:

```text
min sum_k sum_i D_i * ||x_i - mu_k||^2
```

This makes high-demand points more influential in the clustering process.

### Method 3: Proposed Grid-Aware Demand Clustering

Uses spatial, demand, accessibility, and grid-proxy features:

```text
x_i = [lon_i, lat_i, D_i, A_i, G_i]
```

where:

```text
D_i = charging demand score
A_i = accessibility score
G_i = grid risk or grid connection cost proxy
```

This is the proposed method.

The purpose is to generate charging service zones that are not only spatially compact but also demand-aware and grid-aware.

---

## 10. Location-Capacity Optimisation

After clustering, each cluster/service zone should be optimised internally.

Decision variables:

```text
y_j = 1 if candidate site j is selected, otherwise 0
n_j = number of chargers installed at selected site j
x_ij = 1 if demand point i is assigned to station j, otherwise 0
```

Objective function:

```text
min F = w1 * F_distance
      + w2 * F_investment
      + w3 * F_grid
      + w4 * F_queue
```

Where:

### User Distance Cost

```text
F_distance = sum_i sum_j D_i * d_ij * x_ij
```

### Investment Cost

```text
F_investment = sum_j fixed_cost_j * y_j + charger_cost * n_j
```

### Grid Impact Cost

```text
F_grid = sum_j P_j * grid_risk_j
```

### Queue / Capacity Penalty

For the first version, use a simplified capacity penalty:

```text
F_queue = sum_j max(0, assigned_demand_j - station_capacity_j)
```

A more advanced version can later use M/M/s queueing theory.

---

## 11. Baseline Methods

The proposed method should be compared against several baselines:

```text
M1: Random candidate site selection
M2: Existing charging station expansion
M3: Standard geographical K-means
M4: Demand-weighted K-means
M5: Proposed grid-aware demand clustering + optimisation
```

---

## 12. Evaluation Metrics

The project should evaluate the methods using:

```text
1. Average service distance
2. Demand coverage ratio
3. Total investment cost
4. Grid impact score
5. Station utilisation balance
6. Unserved demand
7. Computation time
8. Number of selected charging stations
9. Number of installed chargers
```

The key claim should not only be that the proposed method reduces travel distance, but that it provides a better trade-off among:

```text
user convenience
demand coverage
investment cost
grid impact
station utilisation
```

---

## 13. Expected Figures

The project should generate publication-quality figures:

```text
Figure 1: Overall methodology framework
Figure 2: London case study area and existing EVCS distribution
Figure 3: Charging demand heatmap
Figure 4: Clustering-based service zones
Figure 5: Candidate and selected charging station sites
Figure 6: Comparison of clustering methods
Figure 7: Pareto front or trade-off curve
Figure 8: Sensitivity analysis of number of clusters
```

---

## 14. Expected Tables

The project should generate:

```text
Table 1: Data sources
Table 2: Model parameters
Table 3: Baseline methods
Table 4: Main comparison results
Table 5: Sensitivity analysis
```

---

## 15. Repository Structure

Recommended GitHub repository name:

```text
evcs-clustering-planning
```

Recommended structure:

```text
evcs-clustering-planning/
│
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── .gitignore
│
├── PROJECT_BLUEPRINT.md
│
├── data/
│   ├── raw/
│   │   ├── README.md
│   │   └── .gitkeep
│   ├── interim/
│   │   └── .gitkeep
│   └── processed/
│       └── .gitkeep
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_demand_modelling.ipynb
│   ├── 03_clustering_analysis.ipynb
│   ├── 04_location_optimisation.ipynb
│   └── 05_results_visualisation.ipynb
│
├── src/
│   └── evcs_planning/
│       ├── __init__.py
│       ├── config.py
│       │
│       ├── data/
│       │   ├── __init__.py
│       │   ├── loaders.py
│       │   ├── preprocess.py
│       │   └── spatial.py
│       │
│       ├── demand/
│       │   ├── __init__.py
│       │   ├── features.py
│       │   └── demand_model.py
│       │
│       ├── clustering/
│       │   ├── __init__.py
│       │   ├── kmeans.py
│       │   ├── weighted_kmeans.py
│       │   └── grid_aware_clustering.py
│       │
│       ├── optimisation/
│       │   ├── __init__.py
│       │   ├── candidate_generation.py
│       │   ├── objective.py
│       │   ├── constraints.py
│       │   └── solver.py
│       │
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── metrics.py
│       │   └── baselines.py
│       │
│       └── visualisation/
│           ├── __init__.py
│           ├── maps.py
│           ├── plots.py
│           └── tables.py
│
├── scripts/
│   ├── download_data.py
│   ├── build_demand_dataset.py
│   ├── run_clustering.py
│   ├── run_optimisation.py
│   ├── run_london_case.py
│   └── generate_figures.py
│
├── configs/
│   ├── london_barnet.yaml
│   ├── london_inner.yaml
│   └── default_params.yaml
│
├── results/
│   ├── figures/
│   ├── tables/
│   └── logs/
│
├── paper/
│   ├── main.tex
│   ├── sections/
│   │   ├── 01_introduction.tex
│   │   ├── 02_related_work.tex
│   │   ├── 03_problem_formulation.tex
│   │   ├── 04_methodology.tex
│   │   ├── 05_case_study.tex
│   │   ├── 06_results.tex
│   │   └── 07_conclusion.tex
│   ├── figures/
│   ├── tables/
│   └── references.bib
│
└── docs/
    ├── methodology.md
    ├── data_sources.md
    ├── model_formulation.md
    └── experiment_protocol.md
```

---

## 16. Paper Structure

The paper should follow this structure:

```text
1. Introduction
2. Related Work
3. Problem Formulation
4. Proposed Methodology
5. Case Study and Data Description
6. Results and Discussion
7. Conclusion
```

---

## 17. Introduction Direction

The Introduction should focus on:

```text
1. Rapid EV adoption creates pressure on public charging infrastructure.
2. Charging demand is unevenly distributed across urban areas.
3. Large-scale charging station planning is computationally complex.
4. Existing methods often separate traffic/demand-side planning from grid-side feasibility.
5. Clustering can decompose the problem into interpretable service zones.
6. This paper proposes a grid-aware clustering-based siting and sizing framework.
```

Proposed contribution statements:

```text
1. A grid-aware demand clustering method is proposed to partition urban EV charging demand into service zones.
2. A location-capacity optimisation model is formulated to jointly consider user travel cost, investment cost, demand coverage, and grid impact.
3. A London case study is developed using open urban datasets, and the proposed method is compared with standard clustering and baseline siting strategies.
```

---

## 18. Related Work Structure

The literature review should be organised into:

```text
2.1 EV charging demand modelling
2.2 EV charging station siting and sizing
2.3 Clustering-based infrastructure planning
2.4 Grid-aware EV charging infrastructure planning
```

The key research gap:

> Existing clustering-based EVCS planning methods usually rely on geographical or demand-only features. The interaction between urban charging demand, transport accessibility, and grid hosting capability is not sufficiently embedded into the clustering partition stage.

---

## 19. MVP Scope

The minimum viable version should include:

```text
1. London Barnet or selected London borough case study
2. Open-source spatial data loading
3. Demand point generation
4. Demand-weighted feature construction
5. Three clustering methods
6. Candidate station generation
7. Simplified location-capacity optimisation
8. Baseline comparison
9. Visualisation maps and result tables
10. Initial paper draft sections
```

Do not include in the MVP:

```text
1. Real power flow calculation
2. Full UK distribution network modelling
3. Deep learning demand forecasting
4. Full Greater London large-scale optimisation
5. Web application interface
```

These can be future extensions.

---

## 20. Suggested Development Order

Codex should execute in this order:

### Step 1: Create repository skeleton

Generate all folders and placeholder files.

### Step 2: Create documentation

Create:

```text
README.md
PROJECT_BLUEPRINT.md
docs/methodology.md
docs/data_sources.md
docs/model_formulation.md
docs/experiment_protocol.md
```

### Step 3: Implement data modules

Create loading and preprocessing utilities:

```text
src/evcs_planning/data/loaders.py
src/evcs_planning/data/preprocess.py
src/evcs_planning/data/spatial.py
```

### Step 4: Implement demand model

Create:

```text
src/evcs_planning/demand/features.py
src/evcs_planning/demand/demand_model.py
```

### Step 5: Implement clustering methods

Create:

```text
src/evcs_planning/clustering/kmeans.py
src/evcs_planning/clustering/weighted_kmeans.py
src/evcs_planning/clustering/grid_aware_clustering.py
```

### Step 6: Implement optimisation layer

Create:

```text
src/evcs_planning/optimisation/candidate_generation.py
src/evcs_planning/optimisation/objective.py
src/evcs_planning/optimisation/constraints.py
src/evcs_planning/optimisation/solver.py
```

### Step 7: Implement evaluation and visualisation

Create:

```text
src/evcs_planning/evaluation/metrics.py
src/evcs_planning/evaluation/baselines.py
src/evcs_planning/visualisation/maps.py
src/evcs_planning/visualisation/plots.py
src/evcs_planning/visualisation/tables.py
```

### Step 8: Create run scripts

Create:

```text
scripts/build_demand_dataset.py
scripts/run_clustering.py
scripts/run_optimisation.py
scripts/run_london_case.py
scripts/generate_figures.py
```

### Step 9: Create paper skeleton

Create LaTeX paper structure under:

```text
paper/
```

with section files and placeholder content.

---

## 21. Coding Style Requirements

The code should be:

```text
1. Modular
2. Reproducible
3. Config-driven
4. Documented with docstrings
5. Able to run on synthetic data if real data is not yet downloaded
6. Easy to extend to other cities
```

Important design principle:

> Do not hard-code London-specific logic into the core algorithm. London should be defined through config files and data inputs.

---

## 22. Synthetic Data Support

Because real data collection may take time, the first implementation should include synthetic data generation.

Add a utility that can generate:

```text
1. Synthetic demand points
2. Synthetic candidate sites
3. Synthetic existing charging stations
4. Synthetic grid-risk scores
5. Synthetic accessibility scores
```

This allows the whole pipeline to run before real London data is fully available.

Suggested file:

```text
src/evcs_planning/data/synthetic.py
```

Suggested script:

```text
scripts/run_synthetic_case.py
```

---

## 23. Final Expected GitHub README Content

The README should include:

```text
Project title
Research motivation
Methodology overview
Repository structure
Installation guide
How to run synthetic case
How to run London case
Data sources
Expected outputs
Citation placeholder
License
```

Example command:

```bash
git clone <repo-url>
cd evcs-clustering-planning
pip install -e .
python scripts/run_synthetic_case.py
python scripts/run_london_case.py
```

---

## 24. Long-Term Extensions

Future work can include:

```text
1. Real distribution network topology
2. Power flow constraints
3. Dynamic time-of-day charging demand
4. Queueing theory model
5. Multi-objective optimisation with NSGA-II or MOPSO
6. Shenzhen UrbanEV validation
7. V2G-aware charging infrastructure planning
8. Integration with renewable generation and battery storage
9. Web-based interactive planning dashboard
```

---

## 25. Key Instruction for Codex

Please implement this as a clean research codebase.

First priority:

```text
Build the repository skeleton, documentation, synthetic data pipeline, clustering methods, simplified optimisation model, evaluation metrics, and visualisation outputs.
```

Second priority:

```text
Prepare the London data interface and paper skeleton.
```

Do not overcomplicate the first version with real grid power flow or deep learning. The MVP should run end-to-end with synthetic data and be easily replaceable with real London open data later.

---

## 26. Suggested First Prompt for Codex

Use this as the first implementation prompt:

```text
Create the complete repository skeleton and implement a runnable synthetic-data MVP for the GDC-LCO EV charging station planning project based on PROJECT_BLUEPRINT.md.

The MVP should include:
1. A clean Python package structure under src/evcs_planning.
2. Synthetic demand and candidate-site generation.
3. Geographical K-means, demand-weighted K-means, and grid-aware clustering.
4. A simplified location-capacity optimisation solver.
5. Evaluation metrics and visualisation scripts.
6. A runnable script: scripts/run_synthetic_case.py.
7. Basic documentation and README.
8. A LaTeX paper skeleton under paper/.

Please keep the implementation modular, reproducible, config-driven, and easy to extend to real London data later.
```
