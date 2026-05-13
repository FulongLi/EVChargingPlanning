"""Lightweight NumPy K-means implementation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from evcs_planning.data.spatial import standardize_matrix


@dataclass(frozen=True)
class ClusteringResult:
    labels: np.ndarray
    centers: np.ndarray
    feature_names: list[str]
    inertia: float


def kmeans(
    features: np.ndarray,
    n_clusters: int,
    random_seed: int = 42,
    max_iter: int = 100,
    sample_weight: np.ndarray | None = None,
) -> ClusteringResult:
    """Run K-means with optional sample weights."""
    if n_clusters < 1:
        raise ValueError("n_clusters must be at least 1")
    if len(features) < n_clusters:
        raise ValueError("n_clusters cannot exceed number of samples")

    x = np.asarray(features, dtype=float)
    weights = np.ones(len(x), dtype=float) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    weights = np.maximum(weights, 1e-9)

    rng = np.random.default_rng(random_seed)
    initial_index = rng.choice(len(x), size=n_clusters, replace=False, p=weights / weights.sum())
    centers = x[initial_index].copy()
    labels = np.zeros(len(x), dtype=int)

    for _ in range(max_iter):
        distances = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = distances.argmin(axis=1)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for cluster in range(n_clusters):
            mask = labels == cluster
            if not mask.any():
                centers[cluster] = x[rng.integers(0, len(x))]
                continue
            cluster_weights = weights[mask]
            centers[cluster] = np.average(x[mask], axis=0, weights=cluster_weights)

    inertia = float((((x - centers[labels]) ** 2).sum(axis=1) * weights).sum())
    return ClusteringResult(labels=labels, centers=centers, feature_names=[], inertia=inertia)


def geographical_kmeans(
    demand_points: pd.DataFrame,
    n_clusters: int,
    random_seed: int = 42,
    max_iter: int = 100,
) -> ClusteringResult:
    """Cluster demand points using only longitude and latitude."""
    feature_names = ["lon", "lat"]
    features = standardize_matrix(demand_points[feature_names].to_numpy())
    result = kmeans(features, n_clusters=n_clusters, random_seed=random_seed, max_iter=max_iter)
    return ClusteringResult(result.labels, result.centers, feature_names, result.inertia)
