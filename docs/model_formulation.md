# Model Formulation

Demand points are indexed by `i` and candidate charging station sites are indexed by `j`.

Decision variables:

```text
y_j = 1 if candidate site j is selected
n_j = number of chargers installed at site j
x_ij = 1 if demand point i is assigned to station j
```

The target objective is:

```text
min F = w1 F_distance
      + w2 F_investment
      + w3 F_grid
      + w4 F_capacity
```

where:

- `F_distance` measures demand-weighted service distance.
- `F_investment` measures fixed site cost and charger installation cost.
- `F_grid` measures grid-risk-weighted charging power.
- `F_capacity` penalises assigned demand that exceeds installed capacity.

The MVP implements a greedy approximation. The formulation is documented here so a MILP or multi-objective solver can replace the heuristic later.
