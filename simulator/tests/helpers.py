"""Shared test helpers. All parameter values here are SYNTHETIC test fixtures."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from simulator.params import SimParams, params_from_overrides
from simulator.state import State


def make_params(
    *,
    ratio: float = 9.0,
    review: int = 1,
    mode: str = "lost_sales",
    lead: int | None = None,
    extra: dict[str, Any] | None = None,
) -> SimParams:
    """h = 1 per unit-week (rate 52, price 1), p = ratio. `lead` -> deterministic lead time."""
    overrides: dict[str, Any] = {
        "simulation": {"mode": mode, "review_period_weeks": review},
        "costs": {
            "holding_rate_annual": 52.0,
            "default_unit_price": 1.0,
            "stockout_to_holding_ratio": ratio,
        },
    }
    if lead is not None:
        overrides["lead_time"] = {"distribution": "deterministic", "fixed_weeks": lead}
    for section, values in (extra or {}).items():
        overrides.setdefault(section, {}).update(values)
    return params_from_overrides(overrides)


def stochastic_params(
    *, mode: str = "lost_sales", review: int = 1, shock_prob: float = 0.3
) -> SimParams:
    return make_params(
        mode=mode,
        review=review,
        extra={"lead_time": {"shock_prob": shock_prob, "shock_weeks": 2}},
    )


def make_state(
    t: int,
    pmf: dict[int, float],
    *,
    seed: int = 0,
    series_id: str = "s",
    on_hand: int = 0,
    pipeline: tuple[tuple[int, int], ...] = (),
) -> State:
    return State(
        t=t,
        on_hand=on_hand,
        pipeline=pipeline,
        inventory_position=on_hand + sum(q for _, q in pipeline),
        sales_history=np.zeros(t, dtype=np.int64),
        demand_history=np.zeros(t, dtype=np.int64),
        lead_time_pmf=pmf,
        seed=seed,
        series_id=series_id,
    )


def poisson_pmf(lam: float, n_max: int) -> np.ndarray:
    k = np.arange(n_max + 1)
    log_pmf = k * math.log(lam) - lam - np.array([math.lgamma(x + 1) for x in k])
    return np.exp(log_pmf)


def poisson_quantile(lam: float, tau: float) -> int:
    """Smallest x with Poisson(lam) cdf(x) >= tau."""
    cdf = 0.0
    x = 0
    while True:
        cdf += float(poisson_pmf(lam, x)[x])
        if cdf >= tau:
            return x
        x += 1


def expected_newsvendor_cost(s: int, pmf: np.ndarray, h: float, p: float) -> float:
    """h*E(S-D)+ + p*E(D-S)+ for demand pmf over 0..len-1."""
    d = np.arange(len(pmf))
    return float(np.sum(pmf * (h * np.maximum(s - d, 0) + p * np.maximum(d - s, 0))))
