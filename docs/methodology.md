# Methodology

The project follows the GDC-LCO framework: Grid-aware Demand Clustering for Location-Capacity Optimisation.

The main pipeline is:

```text
Data preparation
  -> demand feature construction
  -> grid-risk and accessibility proxy construction
  -> service-zone clustering
  -> candidate site selection
  -> location-capacity optimisation
  -> evaluation
```

The current experiments compare three partitioning strategies:

1. Geographical K-means using longitude and latitude only.
2. Demand-weighted K-means using demand as sample weights.
3. Grid-aware demand clustering using spatial, demand, accessibility, and grid-risk features.

The current optimisation layer uses a greedy heuristic so the full pipeline can run without external solvers. A MILP or multi-objective optimiser can be added later behind the same module boundary.
