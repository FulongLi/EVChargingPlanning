"""Demand-weighted K-means baseline."""

from __future__ import annotations

import pandas as pd

from evcs_planning.clustering.kmeans import ClusteringResult, kmeans
from evcs_planning.data.spatial import standardize_matrix


def demand_weighted_kmeans(
    demand_points: pd.DataFrame,
    n_clusters: int,
    random_seed: int = 42,
    max_iter: int = 100,
    demand_column: str = "demand",
) -> ClusteringResult:
    """Cluster demand points spatially while weighting centroids by charging demand."""
    feature_names = ["lon", "lat"]
    features = standardize_matrix(demand_points[feature_names].to_numpy())
    result = kmeans(
        features,
        n_clusters=n_clusters,
        random_seed=random_seed,
        max_iter=max_iter,
        sample_weight=demand_points[demand_column].to_numpy(),
    )
    return ClusteringResult(result.labels, result.centers, feature_names, result.inertia)
