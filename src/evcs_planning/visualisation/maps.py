"""SVG map-style scatter plots."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PALETTE = [
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#9333ea",
    "#ea580c",
    "#0891b2",
    "#be123c",
    "#4d7c0f",
    "#7c3aed",
    "#0f766e",
]


def write_cluster_map_svg(
    demand_points: pd.DataFrame,
    labels,
    candidates: pd.DataFrame,
    selected_sites: pd.DataFrame,
    path: str | Path,
    title: str,
    width: int = 980,
    height: int = 680,
) -> None:
    """Write a publication-draft SVG map of clusters and selected stations."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    all_lon = np.concatenate([demand_points["lon"].to_numpy(), candidates["lon"].to_numpy()])
    all_lat = np.concatenate([demand_points["lat"].to_numpy(), candidates["lat"].to_numpy()])
    margin = 54

    def sx(lon):
        return margin + (lon - all_lon.min()) / (all_lon.max() - all_lon.min()) * (width - 2 * margin)

    def sy(lat):
        return height - margin - (lat - all_lat.min()) / (all_lat.max() - all_lat.min()) * (height - 2 * margin)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{margin}" y="34" font-family="Arial" font-size="22" font-weight="700" fill="#111827">{_escape(title)}</text>',
        f'<rect x="{margin}" y="{margin}" width="{width - 2 * margin}" height="{height - 2 * margin}" fill="#f8fafc" stroke="#cbd5e1"/>',
    ]

    demand_size = 2.2 + 4.2 * _safe_scale(demand_points["demand"].to_numpy())
    for index, row in demand_points.reset_index(drop=True).iterrows():
        color = PALETTE[int(labels[index]) % len(PALETTE)]
        parts.append(
            f'<circle cx="{sx(row.lon):.2f}" cy="{sy(row.lat):.2f}" r="{demand_size[index]:.2f}" '
            f'fill="{color}" fill-opacity="0.58" stroke="none"/>'
        )

    for _, row in candidates.iterrows():
        parts.append(
            f'<circle cx="{sx(row.lon):.2f}" cy="{sy(row.lat):.2f}" r="3.1" fill="none" '
            'stroke="#64748b" stroke-width="0.8" stroke-opacity="0.55"/>'
        )

    for _, row in selected_sites.iterrows():
        radius = 6.0 + 0.65 * float(row.get("chargers", 1))
        parts.append(
            f'<circle cx="{sx(row.lon):.2f}" cy="{sy(row.lat):.2f}" r="{radius:.2f}" fill="#111827" '
            'fill-opacity="0.92" stroke="#ffffff" stroke-width="1.8"/>'
        )
        parts.append(
            f'<text x="{sx(row.lon) + radius + 3:.2f}" y="{sy(row.lat) + 4:.2f}" '
            'font-family="Arial" font-size="10" fill="#111827">'
            f'{int(row.get("chargers", 0))}</text>'
        )

    parts.extend(
        [
            f'<text x="{margin}" y="{height - 18}" font-family="Arial" font-size="12" fill="#475569">Demand dots are sized by demand; black stations are selected and labelled by charger count.</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(parts), encoding="utf-8")


def _safe_scale(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    min_value = values.min()
    max_value = values.max()
    if np.isclose(max_value, min_value):
        return np.full_like(values, 0.5)
    return (values - min_value) / (max_value - min_value)


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
