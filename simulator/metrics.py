"""Metrics over the evaluation window (warm-up excluded). Spec section 8."""

from __future__ import annotations

import math

import numpy as np

from simulator.engine import SimResult


def compute_metrics(res: SimResult, skip: int | None = None) -> dict[str, float]:
    """Metrics for weeks skip..T-1 (default skip = the run's warm-up weeks).

    Fill rate and stockout frequency are NaN when undefined (no demand); callers exclude those
    series from averages and log the count. Fill rate is NaN in backorder mode (validation only).
    """
    k = res.warmup_weeks if skip is None else skip
    n = len(res.demand)
    if k >= n:
        raise ValueError(f"skip={k} leaves no evaluation weeks (T={n})")
    sl = slice(k, None)
    demand, sold, lost = res.demand[sl], res.sold[sl], res.lost[sl]
    backlog = res.backlog_end[sl]

    total_demand = int(demand.sum())
    demand_weeks = int((demand > 0).sum())
    short_weeks = int(((lost > 0) | (backlog > 0))[demand > 0].sum())
    backorder = res.mode == "backorder"

    holding = float(res.holding_cost[sl].sum())
    stockout = float(res.stockout_cost[sl].sum())
    order = float(res.order_cost[sl].sum())
    placed = res.ordered[sl] > 0
    eff_lead = (res.arrival_week - np.arange(n))[sl][placed]

    nan = math.nan
    return {
        "total_cost": holding + stockout + order,
        "holding_cost": holding,
        "stockout_cost": stockout,
        "order_cost": order,
        "fill_rate": nan if (total_demand == 0 or backorder) else float(sold.sum()) / total_demand,
        "stockout_frequency": nan if demand_weeks == 0 else short_weeks / demand_weeks,
        "cycle_service_level": nan if demand_weeks == 0 else 1.0 - short_weeks / demand_weeks,
        "avg_inventory": float(np.maximum(res.on_hand_end[sl], 0).mean()),
        "orders_placed": float(placed.sum()),
        "mean_effective_lead_time": float(eff_lead.mean()) if len(eff_lead) else nan,
        "max_effective_lead_time": float(eff_lead.max()) if len(eff_lead) else nan,
        "total_demand": float(total_demand),
        "n_weeks": float(n - k),
    }
