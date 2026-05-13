"""Spatial helper functions for lightweight experiments."""

from __future__ import annotations

import numpy as np


EARTH_RADIUS_KM = 6371.0088


def haversine_km(lon1, lat1, lon2, lat2):
    """Return great-circle distance in kilometres.

    Parameters can be scalars or NumPy-compatible arrays.
    """
    lon1_rad = np.radians(lon1)
    lat1_rad = np.radians(lat1)
    lon2_rad = np.radians(lon2)
    lat2_rad = np.radians(lat2)
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def pairwise_haversine_km(left_lon, left_lat, right_lon, right_lat) -> np.ndarray:
    """Compute pairwise haversine distances between two point sets."""
    left_lon = np.asarray(left_lon)[:, None]
    left_lat = np.asarray(left_lat)[:, None]
    right_lon = np.asarray(right_lon)[None, :]
    right_lat = np.asarray(right_lat)[None, :]
    return haversine_km(left_lon, left_lat, right_lon, right_lat)


def min_max_scale(values, lower: float = 0.0, upper: float = 1.0) -> np.ndarray:
    """Scale values to a fixed interval, handling constant arrays safely."""
    array = np.asarray(values, dtype=float)
    min_value = float(np.nanmin(array))
    max_value = float(np.nanmax(array))
    if np.isclose(max_value, min_value):
        return np.full_like(array, (lower + upper) / 2.0, dtype=float)
    scaled = (array - min_value) / (max_value - min_value)
    return lower + scaled * (upper - lower)


def standardize_matrix(matrix: np.ndarray) -> np.ndarray:
    """Z-score standardise a numeric feature matrix."""
    matrix = np.asarray(matrix, dtype=float)
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std[std == 0.0] = 1.0
    return (matrix - mean) / std
