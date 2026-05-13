"""Baseline method registry."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from evcs_planning.clustering.grid_aware_clustering import grid_aware_demand_clustering
from evcs_planning.clustering.kmeans import ClusteringResult, geographical_kmeans
from evcs_planning.clustering.weighted_kmeans import demand_weighted_kmeans


Clusterer = Callable[..., ClusteringResult]


def get_clustering_methods() -> dict[str, Clusterer]:
    """Return clustering methods used in the MVP comparison."""
    return {
        "geographical_kmeans": geographical_kmeans,
        "demand_weighted_kmeans": demand_weighted_kmeans,
        "grid_aware_demand_clustering": grid_aware_demand_clustering,
    }


def attach_labels(demand_points: pd.DataFrame, labels, label_column: str = "cluster") -> pd.DataFrame:
    """Return a demand table with cluster labels attached."""
    result = demand_points.copy()
    result[label_column] = labels
    return result
