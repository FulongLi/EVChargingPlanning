"""Objective component calculations."""

from __future__ import annotations

import pandas as pd


def station_investment_cost(selected_sites: pd.DataFrame) -> float:
    """Compute total fixed and charger investment cost."""
    if selected_sites.empty:
        return 0.0
    return float((selected_sites["fixed_cost"] + selected_sites["charger_cost"] * selected_sites["chargers"]).sum())


def station_grid_impact(selected_sites: pd.DataFrame, charger_power_kw: float = 50.0) -> float:
    """Compute a simplified grid impact score."""
    if selected_sites.empty:
        return 0.0
    return float((selected_sites["chargers"] * charger_power_kw * selected_sites["grid_risk"]).sum())
