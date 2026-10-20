"""(f) Monotonicity in the stockout cost.

Proves: policy and cost respond in the right direction. Does not prove: levels are optimal.
"""

import numpy as np

from simulator import (
    BaseStock,
    FixedOrderUpTo,
    PoissonPaths,
    compute_metrics,
    effective_pmf,
    simulate,
)
from simulator.tests.helpers import make_params, make_state

RATIOS = [1, 2, 4, 9, 19, 49, 99]


def test_base_stock_level_nondecreasing_in_p():
    pmf = effective_pmf(make_params().lead_time)
    levels = []
    for ratio in RATIOS:
        params = make_params(ratio=ratio)
        policy = BaseStock(params, PoissonPaths(10.0))
        # Identical (seed, series, t) -> identical sample paths and lead draws across ratios.
        levels.append(policy.level(make_state(5, pmf, seed=3, series_id="f")))
    assert all(a <= b for a, b in zip(levels, levels[1:], strict=False)), levels
    assert levels[0] < levels[-1]


def test_lost_units_nonincreasing_in_s_and_argmin_cost_nondecreasing_in_p():
    params = make_params(ratio=1, lead=2)
    grid = list(range(0, 61))
    demand = np.random.default_rng(11).poisson(10, size=1500)

    lost, hold = [], []
    for s in grid:
        res = simulate(demand, FixedOrderUpTo(s), params, seed=0, series_id="f2")
        m = compute_metrics(res, skip=10)
        lost.append(res.lost[10:].sum())
        hold.append(m["holding_cost"])
    assert all(a >= b for a, b in zip(lost, lost[1:], strict=False)), "lost units rose with S"

    # Cost(S; p) = holding(S) + p * lost(S), same demand and lead schedule for every S and p.
    argmins = []
    for p in [1, 2, 4, 9, 19, 49, 99]:
        costs = np.array(hold) + p * np.array(lost)
        argmins.append(int(np.flatnonzero(costs == costs.min()).max()))  # largest argmin
    assert all(a <= b for a, b in zip(argmins, argmins[1:], strict=False)), argmins
