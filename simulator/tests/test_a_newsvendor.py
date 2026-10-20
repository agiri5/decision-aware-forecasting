"""(a) Single-period newsvendor: simulator cost and argmin match the closed form.

Proves: single-period serve logic and cost accounting. Does not prove: lead times or dynamics.
"""

import numpy as np
import pytest

from simulator import ZeroPolicy, compute_metrics, simulate
from simulator.tests.helpers import expected_newsvendor_cost, make_params, poisson_pmf

H, RATIO = 1.0, 9.0


def _sim_expected_cost(s: int, pmf: np.ndarray, params) -> float:
    """Exact expectation: one T=1 simulation per demand value, weighted by the pmf."""
    total = 0.0
    for d, w in enumerate(pmf):
        res = simulate(
            np.array([d]), ZeroPolicy(), params, seed=0, series_id="nv", initial_on_hand=s
        )
        total += w * compute_metrics(res, skip=0)["total_cost"]
    return total


@pytest.mark.parametrize("name", ["poisson10", "uniform0_20"])
def test_newsvendor_matches_closed_form(name):
    params = make_params(ratio=RATIO, lead=1)
    pmf = poisson_pmf(10.0, 80) if name == "poisson10" else np.full(21, 1 / 21)
    p = RATIO * H

    grid = range(0, len(pmf) + 1)
    sim_costs = [_sim_expected_cost(s, pmf, params) for s in grid]
    closed = [expected_newsvendor_cost(s, pmf, H, p) for s in grid]
    np.testing.assert_allclose(sim_costs, closed, rtol=0, atol=1e-9)

    # Optimal stock: smallest S with F(S) >= p/(p+h).
    cdf = np.cumsum(pmf)
    s_star = int(np.argmax(cdf >= p / (p + H)))
    assert int(np.argmin(sim_costs)) == s_star or sim_costs[s_star] == min(sim_costs)
    assert sim_costs[s_star] == pytest.approx(min(closed), abs=1e-9)
