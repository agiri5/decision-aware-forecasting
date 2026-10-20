"""(b) Zero-variance demand: protection-interval timing has no off-by-one.

Proves: the protection interval is exactly L + R weeks of demand (receive/order/serve order).
Does not prove: stochastic behaviour.

Constant demand d, deterministic lead time L, review period R. At S* = d*(L+R):
  - zero lost sales;
  - holding per week averages d*(R-1)/2 (exactly zero for R = 1; with R > 1 stock legitimately
    sits on hand within a review cycle: per cycle sum_i d*(R-1-i) = d*R*(R-1)/2).
At S*+1: same, plus exactly one extra unit on hand every week.
At S*-1: shortage appears (R = 1: exactly one lost unit every L+R weeks in lost-sales mode,
exactly one backlogged unit every week in backorder mode).
"""

import numpy as np
import pytest

from simulator import (
    BaseStock,
    FixedOrderUpTo,
    OraclePaths,
    compute_metrics,
    simulate,
)
from simulator.tests.helpers import make_params

D = 5
SKIP = 12  # even, so the evaluation window spans whole review cycles for R in {1, 2}
T = SKIP + 40


def _run(policy, params, d=D):
    demand = np.full(T, d)
    res = simulate(demand, policy, params, seed=0, series_id="zv")
    return res, compute_metrics(res, skip=SKIP)


@pytest.mark.parametrize("lead", [1, 2, 4])
@pytest.mark.parametrize("review", [1, 2])
def test_exact_protection_interval(lead, review):
    params = make_params(lead=lead, review=review)
    s_star = D * (lead + review)
    weeks = T - SKIP
    extra_holding = D * (review - 1) / 2 * weeks  # h = 1

    # Base-stock fed constant paths computes S* exactly.
    res, m = _run(BaseStock(params, OraclePaths(np.full(T, D))), params)
    levels = res.order_up_to[::review]
    assert np.all(levels == s_star)
    assert m["stockout_cost"] == 0.0 and res.lost.sum() == 0
    assert m["holding_cost"] == pytest.approx(extra_holding)

    # One unit above: one extra unit on hand every week, still no losses.
    res, m = _run(FixedOrderUpTo(s_star + 1), params)
    assert res.lost.sum() == 0
    assert m["holding_cost"] == pytest.approx(extra_holding + weeks)

    # One unit below: shortage appears.
    res, m = _run(FixedOrderUpTo(s_star - 1), params)
    lost = int(res.lost[SKIP:].sum())
    assert lost > 0
    if review == 1:
        assert abs(lost - weeks / (lead + review)) <= 1
        assert m["holding_cost"] == 0.0


@pytest.mark.parametrize("lead", [1, 2, 4])
def test_backorder_mode_one_short(lead):
    params = make_params(lead=lead, review=1, mode="backorder")
    s_star = D * (lead + 1)
    weeks = T - SKIP

    _, m = _run(FixedOrderUpTo(s_star), params)
    assert m["total_cost"] == 0.0

    # One unit short: exactly one backlogged unit every week, priced at p = 9 per unit-week.
    _, m = _run(FixedOrderUpTo(s_star - 1), params)
    assert m["stockout_cost"] == pytest.approx(9.0 * weeks)
    assert m["holding_cost"] == 0.0
