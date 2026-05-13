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
    title: str = "Planning Result",
    subtitle: str = "",
    note: str = "",
    boundary: pd.DataFrame | None = None,
    width: int = 1200,
    height: int = 900,
) -> None:
    """Write a raster map-style PNG without external plotting dependencies."""
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    _draw_text(image, 64, 24, title, [15, 23, 42], scale=3)
    if subtitle:
        _draw_text(image, 64, 58, subtitle, [71, 85, 105], scale=2)
    left, right, top, bottom = 96, width - 72, 104, height - 100
    _rect(image, left, top, right, bottom, [248, 250, 252])
    _rect_outline(image, left, top, right, bottom, [148, 163, 184])

    all_lon = np.concatenate([demand_points["lon"].to_numpy(), candidates["lon"].to_numpy()])
    all_lat = np.concatenate([demand_points["lat"].to_numpy(), candidates["lat"].to_numpy()])
    if boundary is not None and not boundary.empty:
        all_lon = np.concatenate([all_lon, boundary["lon"].to_numpy()])
        all_lat = np.concatenate([all_lat, boundary["lat"].to_numpy()])
    lon_min, lon_max = float(all_lon.min()), float(all_lon.max())
    lat_min, lat_max = float(all_lat.min()), float(all_lat.max())

    def sx(lon):
        return int(left + (lon - lon_min) / max(lon_max - lon_min, 1e-9) * (right - left))

    def sy(lat):
        return int(bottom - (lat - lat_min) / max(lat_max - lat_min, 1e-9) * (bottom - top))

    _draw_coordinate_ticks(image, lon_min, lon_max, lat_min, lat_max, sx, sy, left, right, top, bottom)

    if boundary is not None and not boundary.empty:
        for _, part in boundary.groupby("part"):
            coords = [(sx(row.lon), sy(row.lat)) for row in part.itertuples()]
            _polyline(image, coords, [15, 23, 42], thickness=2)

    demand_size = 2.0 + 5.0 * _safe_scale(demand_points["demand"].to_numpy())
    for index, row in demand_points.reset_index(drop=True).iterrows():
        color = PALETTE[int(labels[index]) % len(PALETTE)]
        _circle(image, sx(row.lon), sy(row.lat), int(demand_size[index]), color, alpha=0.60)

    for _, row in candidates.iterrows():
        _circle_outline(image, sx(row.lon), sy(row.lat), 3, [100, 116, 139])

    for _, row in selected_sites.iterrows():
        radius = int(7 + 0.45 * float(row.get("chargers", 1)))
        _circle(image, sx(row.lon), sy(row.lat), radius, [17, 24, 39], alpha=0.95)
        _circle_outline(image, sx(row.lon), sy(row.lat), radius + 1, [255, 255, 255])

    legend_y = height - 72
    _circle(image, 84, legend_y, 5, PALETTE[0], alpha=0.75)
    _draw_text(image, 98, legend_y - 7, "DEMAND POINT / CLUSTER", [51, 65, 85], scale=2)
    _circle_outline(image, 350, legend_y, 5, [100, 116, 139])
    _draw_text(image, 364, legend_y - 7, "CANDIDATE SITE", [51, 65, 85], scale=2)
    _circle(image, 570, legend_y, 8, [17, 24, 39], alpha=0.95)
    _draw_text(image, 588, legend_y - 7, "SELECTED STATION", [51, 65, 85], scale=2)
    if boundary is not None and not boundary.empty:
        _line(image, 835, legend_y, 885, legend_y, [15, 23, 42], thickness=2)
        _draw_text(image, 898, legend_y - 7, "BARNET BOUNDARY", [51, 65, 85], scale=2)
    if note:
        _draw_text(image, 64, height - 36, note, [71, 85, 105], scale=2)
    _write_png(path, image)


def write_metric_bars_png(
    table: pd.DataFrame,
    metric: str,
    path: str | Path,
    title: str = "Metric Comparison",
    note: str = "Longer bars indicate larger metric values.",
    width: int = 1000,
    height: int = 520,
) -> None:
    """Write a simple PNG horizontal bar chart."""
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    max_value = max(float(table[metric].max()), 1e-9)
    left = 260
    _draw_text(image, 40, 24, title, [15, 23, 42], scale=3)
    _draw_text(image, 40, 56, note, [71, 85, 105], scale=2)
    top = 104
    bar_height = 46
    gap = 40
    plot_width = width - left - 90
    colors = [[37, 99, 235], [22, 163, 74], [234, 88, 12], [147, 51, 234]]
    for index, (_, row) in enumerate(table.iterrows()):
        y = top + index * (bar_height + gap)
        bar_width = int(plot_width * float(row[metric]) / max_value)
        _draw_text(image, 40, y + 12, _label(str(row["method"])), [51, 65, 85], scale=2)
        _rect(image, left, y, left + plot_width, y + bar_height, [226, 232, 240])
        _rect(image, left, y, left + bar_width, y + bar_height, colors[index % len(colors)])
        _rect_outline(image, left, y, left + plot_width, y + bar_height, [203, 213, 225])
        _draw_text(image, left + bar_width + 10, y + 12, f"{float(row[metric]):.3f}", [15, 23, 42], scale=2)
    _write_png(path, image)


def write_sensitivity_heatmap_png(
    table: pd.DataFrame,
    row_column: str,
    column_column: str,
    value_column: str,
    path: str | Path,
    title: str = "Sensitivity Analysis",
    note: str = "Rows: number of clusters K. Columns: grid-risk feature weight. Darker red means larger value.",
    width: int = 900,
    height: int = 620,
) -> None:
    """Write a simple raster heatmap for sensitivity-analysis tables."""
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    rows = sorted(table[row_column].unique())
    cols = sorted(table[column_column].unique())
    matrix = np.full((len(rows), len(cols)), np.nan)
    for r_index, row_value in enumerate(rows):
        for c_index, col_value in enumerate(cols):
            values = table.loc[(table[row_column] == row_value) & (table[column_column] == col_value), value_column]
            if not values.empty:
                matrix[r_index, c_index] = float(values.iloc[0])

    valid = matrix[np.isfinite(matrix)]
    min_value = float(valid.min()) if len(valid) else 0.0
    max_value = float(valid.max()) if len(valid) else 1.0
    _draw_text(image, 40, 24, title, [15, 23, 42], scale=3)
    _draw_text(image, 40, 56, note, [71, 85, 105], scale=2)
    left, top = 110, 110
    cell_w = max(1, int((width - left - 80) / max(1, len(cols))))
    cell_h = max(1, int((height - top - 80) / max(1, len(rows))))
    for r_index in range(len(rows)):
        for c_index in range(len(cols)):
            value = matrix[r_index, c_index]
            scaled = 0.5 if not np.isfinite(value) or np.isclose(max_value, min_value) else (value - min_value) / (max_value - min_value)
            color = _heat_color(scaled)
            x0 = left + c_index * cell_w
            y0 = top + r_index * cell_h
            _rect(image, x0, y0, x0 + cell_w - 2, y0 + cell_h - 2, color)
            _draw_text(image, x0 + 6, y0 + 8, f"{value:.2f}", [15, 23, 42], scale=1)
    for r_index, row_value in enumerate(rows):
        _draw_text(image, 52, top + r_index * cell_h + cell_h // 2 - 5, f"K={row_value}", [51, 65, 85], scale=2)
    for c_index, col_value in enumerate(cols):
        _draw_text(image, left + c_index * cell_w + 6, top - 24, f"{col_value:.2f}", [51, 65, 85], scale=1)
    _rect_outline(image, left, top, left + len(cols) * cell_w, top + len(rows) * cell_h, [100, 116, 139])
    _write_png(path, image)


def write_tradeoff_scatter_png(
    table: pd.DataFrame,
    x_metric: str,
    y_metric: str,
    path: str | Path,
    title: str = "Planning Trade-off",
    note: str = "Each point is one method. Lower-left is better when both metrics are costs.",
    width: int = 1200,
    height: int = 680,
) -> None:
    """Write a labelled scatter plot for method trade-offs."""
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    _draw_text(image, 48, 24, title, [15, 23, 42], scale=3)
    _draw_text(image, 48, 56, note, [71, 85, 105], scale=2)
    left, right, top, bottom = 120, width - 70, 105, height - 95
    _rect(image, left, top, right, bottom, [248, 250, 252])
    _rect_outline(image, left, top, right, bottom, [100, 116, 139])
    x_values = table[x_metric].to_numpy(dtype=float)
    y_values = table[y_metric].to_numpy(dtype=float)
    x_min, x_max = float(x_values.min()), float(x_values.max())
    y_min, y_max = float(y_values.min()), float(y_values.max())
    x_pad = max((x_max - x_min) * 0.12, 1e-6)
    y_pad = max((y_max - y_min) * 0.12, 1e-6)

    def sx(value):
        return int(left + (value - x_min + x_pad) / (x_max - x_min + 2 * x_pad) * (right - left))

    def sy(value):
        return int(bottom - (value - y_min + y_pad) / (y_max - y_min + 2 * y_pad) * (bottom - top))

    colors = [[37, 99, 235], [22, 163, 74], [220, 38, 38], [147, 51, 234], [234, 88, 12], [8, 145, 178]]
    for index, (_, row) in enumerate(table.iterrows()):
        color = colors[index % len(colors)]
        x = sx(float(row[x_metric]))
        y = sy(float(row[y_metric]))
        label = _label(str(row["method"]), max_len=20)
        _circle(image, x, y, 10, color, alpha=0.9)
        _circle_outline(image, x, y, 11, [255, 255, 255])
        text_x = x + 16
        text_y = y - 8 + (index % 3) * 18
        if x > right - 250:
            text_x = max(left + 8, x - 16 - len(label) * 12)
        _draw_text(image, text_x, text_y, label, [15, 23, 42], scale=2)

    _draw_text(image, left, bottom + 22, x_metric.replace("_", " ").upper(), [51, 65, 85], scale=2)
    _draw_text(image, 24, top + 20, y_metric.replace("_", " ").upper(), [51, 65, 85], scale=2)
    _write_png(path, image)


def write_method_flowchart_png(path: str | Path, width: int = 1400, height: int = 520) -> None:
    """Write a clean methodology flowchart as PNG."""
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    _draw_text(image, 48, 28, "GDC-LCO METHOD WORKFLOW", [15, 23, 42], scale=3)
    boxes = [
        ("OPEN DATA", "OSM FEATURES\nBOUNDARY\nCONFIG"),
        ("FEATURES", "DEMAND\nACCESS\nGRID RISK"),
        ("CLUSTERING", "GEO KMEANS\nDEMAND KMEANS\nGRID AWARE"),
        ("CANDIDATES", "PARKING\nEXISTING EVCS\nHOTSPOTS"),
        ("OPTIMISATION", "SELECT SITES\nSIZE CHARGERS\nASSIGN DEMAND"),
        ("EVALUATION", "DISTANCE\nCOVERAGE\nGRID IMPACT"),
    ]
    x0, y0 = 48, 130
    box_w, box_h, gap = 190, 170, 34
    for index, (title, body) in enumerate(boxes):
        x = x0 + index * (box_w + gap)
        _rect(image, x, y0, x + box_w, y0 + box_h, [248, 250, 252])
        _rect_outline(image, x, y0, x + box_w, y0 + box_h, [100, 116, 139])
        _draw_text(image, x + 16, y0 + 18, title, [15, 23, 42], scale=2)
        for line_index, line in enumerate(body.split("\n")):
            _draw_text(image, x + 16, y0 + 60 + line_index * 28, line, [71, 85, 105], scale=2)
        if index < len(boxes) - 1:
            y_mid = y0 + box_h // 2
            _line(image, x + box_w + 6, y_mid, x + box_w + gap - 8, y_mid, [37, 99, 235], thickness=3)
            _triangle(image, x + box_w + gap - 8, y_mid, [37, 99, 235])
    _draw_text(image, 48, 350, "CLUSTERING DECOMPOSES THE CITY INTO SERVICE ZONES; OPTIMISATION THEN SELECTS AND SIZES STATIONS WITHIN ZONES.", [51, 65, 85], scale=2)
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


def _draw_coordinate_ticks(image, lon_min, lon_max, lat_min, lat_max, sx, sy, left, right, top, bottom) -> None:
    for value in np.linspace(lon_min, lon_max, 5):
        x = sx(value)
        _line(image, x, bottom, x, bottom + 6, [100, 116, 139])
        _draw_text(image, x - 26, bottom + 14, f"{value:.2f}", [71, 85, 105], scale=1)
    for value in np.linspace(lat_min, lat_max, 5):
        y = sy(value)
        _line(image, left - 6, y, left, y, [100, 116, 139])
        _draw_text(image, 28, y - 4, f"{value:.2f}", [71, 85, 105], scale=1)
    _draw_text(image, (left + right) // 2 - 28, bottom + 36, "LONGITUDE", [51, 65, 85], scale=1)
    _draw_text(image, 24, top - 22, "LAT", [51, 65, 85], scale=1)


def _polyline(image, coords: list[tuple[int, int]], color, thickness: int = 1) -> None:
    for first, second in zip(coords, coords[1:]):
        _line(image, first[0], first[1], second[0], second[1], color, thickness=thickness)


def _line(image, x0: int, y0: int, x1: int, y1: int, color, thickness: int = 1) -> None:
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        _rect(image, x - thickness // 2, y - thickness // 2, x + thickness // 2 + 1, y + thickness // 2 + 1, color)
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy


def _triangle(image, x: int, y: int, color) -> None:
    for offset in range(8):
        _line(image, x - offset, y - offset, x - offset, y + offset, color, thickness=1)


def _heat_color(value: float) -> list[int]:
    value = float(np.clip(value, 0.0, 1.0))
    if value < 0.5:
        t = value / 0.5
        return [int(37 + t * (250 - 37)), int(99 + t * (204 - 99)), int(235 + t * (21 - 235))]
    t = (value - 0.5) / 0.5
    return [int(250 + t * (220 - 250)), int(204 + t * (38 - 204)), int(21 + t * (38 - 21))]


def _label(text: str, max_len: int = 26) -> str:
    replacements = {
        "london_geographical_kmeans": "GEO KMEANS",
        "london_demand_weighted_kmeans": "DEMAND KMEANS",
        "london_grid_aware_demand_clustering": "GRID AWARE",
        "london_demand_hotspot_baseline": "HOTSPOT BASELINE",
        "london_existing_expansion_baseline": "EXISTING BASELINE",
        "london_random_candidate_baseline": "RANDOM BASELINE",
        "ablation_spatial_only": "SPATIAL ONLY",
        "ablation_spatial_demand_weighted": "DEMAND WEIGHTED",
        "ablation_spatial_demand_feature": "DEMAND FEATURE",
        "ablation_plus_accessibility": "PLUS ACCESS",
        "ablation_plus_grid_risk": "PLUS GRID RISK",
    }
    if text in replacements:
        return replacements[text][:max_len]
    text = text.replace("london_", "").replace("_", " ").upper()
    return text[:max_len]


def _draw_text(image, x: int, y: int, text: str, color, scale: int = 2) -> None:
    cursor = x
    for char in text.upper():
        if char == "\n":
            y += 8 * scale
            cursor = x
            continue
        glyph = FONT.get(char, FONT.get("?"))
        if glyph is None:
            cursor += 4 * scale
            continue
        for row_index, row in enumerate(glyph):
            for col_index, value in enumerate(row):
                if value == "1":
                    _rect(
                        image,
                        cursor + col_index * scale,
                        y + row_index * scale,
                        cursor + (col_index + 1) * scale,
                        y + (row_index + 1) * scale,
                        color,
                    )
        cursor += (len(glyph[0]) + 1) * scale


FONT = {
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01111", "10000", "10000", "10000", "10000", "10000", "01111"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "G": ["01111", "10000", "10000", "10011", "10001", "10001", "01111"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["11111", "00100", "00100", "00100", "00100", "00100", "11111"],
    "J": ["00111", "00010", "00010", "00010", "10010", "10010", "01100"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "Q": ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "W": ["10001", "10001", "10001", "10101", "10101", "10101", "01010"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "10000", "11110", "00001", "00001", "11110"],
    "6": ["01110", "10000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00001", "01110"],
    " ": ["0", "0", "0", "0", "0", "0", "0"],
    ".": ["0", "0", "0", "0", "0", "011", "011"],
    ",": ["0", "0", "0", "0", "011", "011", "010"],
    ":": ["0", "011", "011", "0", "011", "011", "0"],
    "-": ["0", "0", "0", "1111", "0", "0", "0"],
    "/": ["00001", "00010", "00010", "00100", "01000", "01000", "10000"],
    "(": ["001", "010", "100", "100", "100", "010", "001"],
    ")": ["100", "010", "001", "001", "001", "010", "100"],
    "%": ["11001", "11010", "00010", "00100", "01000", "01011", "10011"],
    "=": ["0", "11111", "0", "11111", "0", "0", "0"],
    "+": ["0", "00100", "00100", "11111", "00100", "00100", "0"],
    ";": ["0", "011", "011", "0", "011", "011", "010"],
    "?": ["01110", "10001", "00001", "00010", "00100", "0", "00100"],
}


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
