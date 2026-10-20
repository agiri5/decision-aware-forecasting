"""(e) Base-stock cross-check against the closed form (backorder mode, deterministic lead time).

With iid Poisson(lam) demand, R = 1, lead time L and order-up-to S, ending net inventory in
week t+L is S - D_{L+1}, D_{L+1} ~ Poisson((L+1)*lam). Expected cost per week is
h*E(S-D)+ + p*E(D-S)+.

Proves: multi-period pipeline, inventory position and timing agree with theory.
Does not prove: lost-sales optimality (no closed form); lost-sales mode is covered by (a)-(d), (f).
Stockpyl (MIT) was not added; the closed form is the reference. Total simulated volume is
200,000 weeks as in the spec, split into 200 independent runs of 1,000 weeks so the standard
error comes from iid runs and the per-step history copies stay cheap.
"""

import numpy as np
import pytest

from simulator import FixedOrderUpTo, compute_metrics, simulate
from simulator.tests.helpers import expected_newsvendor_cost, make_params, poisson_pmf

LAM, LEAD, RATIO = 4.0, 2, 9.0
N_RUNS, T, SKIP = 200, 1000, 10


def _mean_cost_per_week(s: int) -> tuple[float, float]:
    params = make_params(ratio=RATIO, lead=LEAD, mode="backorder")
    rng = np.random.default_rng(2026)
    per_run = []
    for i in range(N_RUNS):
        demand = rng.poisson(LAM, size=T)
        res = simulate(demand, FixedOrderUpTo(s), params, seed=i, series_id="e")
        m = compute_metrics(res, skip=SKIP)
        per_run.append(m["total_cost"] / m["n_weeks"])
    arr = np.array(per_run)
    return float(arr.mean()), float(arr.std(ddof=1) / np.sqrt(len(arr)))


def test_cost_matches_closed_form_and_optimal_s():
    pmf = poisson_pmf((LEAD + 1) * LAM, 120)
    cdf = np.cumsum(pmf)
    s_star = int(np.argmax(cdf >= RATIO / (RATIO + 1)))
    closed = {s: expected_newsvendor_cost(s, pmf, 1.0, RATIO) for s in range(0, 40)}
    assert min(closed, key=closed.get) == s_star  # closed-form optimum is the critical-ratio S

    for s in (s_star - 3, s_star, s_star + 3):
        mean, se = _mean_cost_per_week(s)
        assert abs(mean - closed[s]) <= 3 * se, (s, mean, closed[s], se)

    # Simulated cost is lowest at the critical-ratio S among its neighbours.
    sims = {s: _mean_cost_per_week(s)[0] for s in (s_star - 3, s_star, s_star + 3)}
    assert sims[s_star] == pytest.approx(min(sims.values()))
