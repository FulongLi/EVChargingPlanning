# Experiment Protocol

The synthetic experiment is run with:

```bash
python3 scripts/run_synthetic_case.py
```

The synthetic script:

1. Generates synthetic demand points, candidate sites, and existing stations.
2. Constructs rule-based charging demand features.
3. Runs geographical K-means, demand-weighted K-means, and grid-aware demand clustering.
4. Solves a simplified location-capacity planning problem for each clustering output.
5. Writes result tables and PNG figures.

The London/Barnet script is:

```bash
python3 scripts/run_london_case.py
```

It downloads OpenStreetMap features through Overpass when `data/raw/barnet_osm_overpass.json` is absent, then builds real-data demand, candidate, existing-station, clustering, optimisation, and result tables.

Primary metrics:

- average service distance
- demand coverage ratio
- total investment cost
- grid impact score
- grid impact per installed charger
- station utilisation mean and standard deviation
- unserved demand
- selected station count
- installed chargers
- computation time

Sensitivity analysis should vary at least the number of clusters and the grid-risk feature weight.
