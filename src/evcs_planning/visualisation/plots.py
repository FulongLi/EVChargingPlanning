"""SVG chart utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_metric_bars_svg(
    table: pd.DataFrame,
    metric: str,
    path: str | Path,
    title: str,
    width: int = 860,
    height: int = 420,
) -> None:
    """Write a compact horizontal bar chart as SVG."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = table[["method", metric]].copy()
    max_value = float(data[metric].max()) if len(data) else 1.0
    max_value = max(max_value, 1e-9)
    left = 230
    top = 72
    bar_height = 34
    gap = 22
    plot_width = width - left - 72

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="40" y="36" font-family="Arial" font-size="22" font-weight="700" fill="#111827">{_escape(title)}</text>',
    ]

    for index, row in data.iterrows():
        y = top + index * (bar_height + gap)
        value = float(row[metric])
        bar_width = plot_width * value / max_value
        parts.append(f'<text x="40" y="{y + 23}" font-family="Arial" font-size="14" fill="#334155">{_escape(str(row.method))}</text>')
        parts.append(f'<rect x="{left}" y="{y}" width="{plot_width}" height="{bar_height}" fill="#e2e8f0"/>')
        parts.append(f'<rect x="{left}" y="{y}" width="{bar_width:.2f}" height="{bar_height}" fill="#2563eb"/>')
        parts.append(f'<text x="{left + bar_width + 8:.2f}" y="{y + 23}" font-family="Arial" font-size="13" fill="#111827">{value:.3f}</text>')

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
