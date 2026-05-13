"""Proposed grid-aware demand clustering method."""

from __future__ import annotations

import numpy as np
import pandas as pd

from evcs_planning.clustering.kmeans import ClusteringResult, kmeans
from evcs_planning.data.spatial import standardize_matrix


def grid_aware_demand_clustering(
    demand_points: pd.DataFrame,
    n_clusters: int,
    random_seed: int = 42,
    max_iter: int = 100,
    feature_weights: dict[str, float] | None = None,
) -> ClusteringResult:
    """Cluster demand points using spatial, demand, accessibility, and grid-risk features."""
    weights = {
        "spatial": 1.0,
        "demand": 0.65,
        "accessibility": 0.4,
        "grid_risk": 0.75,
    }
    if feature_weights:
        weights.update({key: float(value) for key, value in feature_weights.items()})

    feature_names = ["lon", "lat", "demand", "accessibility", "grid_risk"]
    features = standardize_matrix(demand_points[feature_names].to_numpy())
    feature_scale = np.array(
        [
            weights["spatial"],
            weights["spatial"],
            weights["demand"],
            weights["accessibility"],
            weights["grid_risk"],
        ],
        dtype=float,
    )
    weighted_features = features * feature_scale
    result = kmeans(
        weighted_features,
        n_clusters=n_clusters,
        random_seed=random_seed,
        max_iter=max_iter,
        sample_weight=demand_points["demand"].to_numpy(),
    )
    return ClusteringResult(result.labels, result.centers, feature_names, result.inertia)
