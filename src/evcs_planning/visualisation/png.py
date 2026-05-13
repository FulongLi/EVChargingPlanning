"""Dependency-free PNG plotting helpers."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np
import pandas as pd


PALETTE = np.array(
    [
        [37, 99, 235],
        [220, 38, 38],
        [22, 163, 74],
        [147, 51, 234],
        [234, 88, 12],
        [8, 145, 178],
        [190, 18, 60],
        [77, 124, 15],
        [124, 58, 237],
        [15, 118, 110],
    ],
    dtype=np.uint8,
)


def write_cluster_map_png(
    demand_points: pd.DataFrame,
    labels,
    candidates: pd.DataFrame,
    selected_sites: pd.DataFrame,
    path: str | Path,
    width: int = 1200,
    height: int = 840,
) -> None:
    """Write a raster map-style PNG without external plotting dependencies."""
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    _rect(image, 64, 48, width - 64, height - 64, [248, 250, 252])
    _rect_outline(image, 64, 48, width - 64, height - 64, [203, 213, 225])

    all_lon = np.concatenate([demand_points["lon"].to_numpy(), candidates["lon"].to_numpy()])
    all_lat = np.concatenate([demand_points["lat"].to_numpy(), candidates["lat"].to_numpy()])

    def sx(lon):
        return int(64 + (lon - all_lon.min()) / max(all_lon.max() - all_lon.min(), 1e-9) * (width - 128))

    def sy(lat):
        return int(height - 64 - (lat - all_lat.min()) / max(all_lat.max() - all_lat.min(), 1e-9) * (height - 128))

    demand_size = 2.0 + 5.0 * _safe_scale(demand_points["demand"].to_numpy())
    for index, row in demand_points.reset_index(drop=True).iterrows():
        color = PALETTE[int(labels[index]) % len(PALETTE)]
        _circle(image, sx(row.lon), sy(row.lat), int(demand_size[index]), color, alpha=0.60)

    for _, row in candidates.iterrows():
        _circle_outline(image, sx(row.lon), sy(row.lat), 4, [100, 116, 139])

    for _, row in selected_sites.iterrows():
        radius = int(7 + 0.45 * float(row.get("chargers", 1)))
        _circle(image, sx(row.lon), sy(row.lat), radius, [17, 24, 39], alpha=0.95)
        _circle_outline(image, sx(row.lon), sy(row.lat), radius + 1, [255, 255, 255])

    _write_png(path, image)


def write_metric_bars_png(
    table: pd.DataFrame,
    metric: str,
    path: str | Path,
    width: int = 1000,
    height: int = 520,
) -> None:
    """Write a simple PNG horizontal bar chart."""
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    max_value = max(float(table[metric].max()), 1e-9)
    left = 260
    top = 70
    bar_height = 46
    gap = 40
    plot_width = width - left - 90
    colors = [[37, 99, 235], [22, 163, 74], [234, 88, 12], [147, 51, 234]]
    for index, (_, row) in enumerate(table.iterrows()):
        y = top + index * (bar_height + gap)
        bar_width = int(plot_width * float(row[metric]) / max_value)
        _rect(image, left, y, left + plot_width, y + bar_height, [226, 232, 240])
        _rect(image, left, y, left + bar_width, y + bar_height, colors[index % len(colors)])
        _rect_outline(image, left, y, left + plot_width, y + bar_height, [203, 213, 225])
    _write_png(path, image)


def _safe_scale(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if np.isclose(values.max(), values.min()):
        return np.full_like(values, 0.5)
    return (values - values.min()) / (values.max() - values.min())


def _rect(image, x0, y0, x1, y1, color) -> None:
    image[max(0, y0) : min(image.shape[0], y1), max(0, x0) : min(image.shape[1], x1)] = color


def _rect_outline(image, x0, y0, x1, y1, color) -> None:
    _rect(image, x0, y0, x1, y0 + 1, color)
    _rect(image, x0, y1 - 1, x1, y1, color)
    _rect(image, x0, y0, x0 + 1, y1, color)
    _rect(image, x1 - 1, y0, x1, y1, color)


def _circle(image, cx, cy, radius, color, alpha=1.0) -> None:
    y_min, y_max = max(0, cy - radius), min(image.shape[0], cy + radius + 1)
    x_min, x_max = max(0, cx - radius), min(image.shape[1], cx + radius + 1)
    yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
    base = image[y_min:y_max, x_min:x_max]
    color_array = np.asarray(color, dtype=float)
    base[mask] = (alpha * color_array + (1.0 - alpha) * base[mask]).astype(np.uint8)


def _circle_outline(image, cx, cy, radius, color) -> None:
    y_min, y_max = max(0, cy - radius - 1), min(image.shape[0], cy + radius + 2)
    x_min, x_max = max(0, cx - radius - 1), min(image.shape[1], cx + radius + 2)
    yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
    dist = (xx - cx) ** 2 + (yy - cy) ** 2
    mask = (dist <= (radius + 1) ** 2) & (dist >= max(0, radius - 1) ** 2)
    image[y_min:y_max, x_min:x_max][mask] = color


def _write_png(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width, _ = image.shape
    raw = b"".join(b"\x00" + image[row].tobytes() for row in range(height))
    png = b"\x89PNG\r\n\x1a\n"
    png += _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += _chunk(b"IDAT", zlib.compress(raw, level=6))
    png += _chunk(b"IEND", b"")
    path.write_bytes(png)


def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
