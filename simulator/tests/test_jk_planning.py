"""(j) Planning Monte Carlo and effective lead-time pmf; (k) no summed per-horizon quantiles."""

import numpy as np
import pytest

from simulator import BaseStock, PoissonPaths, effective_pmf
from simulator.leadtime import lead_time_schedule
from simulator.params import load_raw
from simulator.tests.helpers import make_params, make_state, poisson_pmf, poisson_quantile


def test_effective_pmf_properties():
    params = make_params()
    lt = params.lead_time
    pmf = effective_pmf(lt)
    assert sum(pmf.values()) == pytest.approx(1.0)
    assert min(pmf) >= lt.min_weeks and max(pmf) <= lt.max_weeks

    draws, _ = lead_time_schedule(lt, 200_000, 0, "pmf-check")
    eff_mean = sum(e * q for e, q in pmf.items())
    assert eff_mean >= draws.mean() - 0.01  # no-overtaking max can only lengthen lead times
    assert eff_mean > draws.mean() + 0.02  # and with shocks it visibly does


def test_deterministic_pmf_is_point_mass():
    assert effective_pmf(make_params(lead=3).lead_time) == {3: 1.0}


def test_mc_error_bounded_at_high_tau_deterministic_lead():
    params = make_params(ratio=99, lead=2)  # tau = 0.99, default n_paths
    pmf = effective_pmf(params.lead_time)
    levels = [
        BaseStock(params, PoissonPaths(10.0)).level(make_state(4, pmf, seed=s, series_id="mc"))
        for s in range(5)
    ]
    assert max(levels) - min(levels) <= 1, levels


def _max_spread(params, origins=range(20)):
    """Largest max-min of S_t across 5 planning seeds, default (stochastic) lead time."""
    pmf = effective_pmf(params.lead_time)
    worst = 0
    for t in origins:
        levels = [
            BaseStock(params, PoissonPaths(10.0)).level(make_state(t, pmf, seed=s, series_id="mc"))
            for s in range(5)
        ]
        worst = max(worst, max(levels) - min(levels))
    return worst


def test_mc_error_bounded_core_arm_operational_lead():
    """(j)(3) core arm: tau = 0.9, default N and lead-time config (G1 finding 1)."""
    assert _max_spread(make_params(ratio=9)) <= 1


def test_mc_error_bounded_medtech_arm_operational_lead():
    """(j)(3) medtech arm: tau = 0.99 with params.yaml's medtech N (G1 finding 1, D11).

    The deterministic variant above cannot trigger the spec's "raise N" rule because the
    point-mass pmf adds no lead-time noise.
    """
    n_medtech = load_raw()["medtech_analog"]["n_paths"]
    params = make_params(ratio=99, extra={"policy": {"n_paths": n_medtech}})
    assert _max_spread(params) <= 1


def test_pooled_level_matches_exact_mixture_quantile():
    """The pooled estimator targets the mixture over E of Poisson(10 (E + R)) sums."""
    params = make_params(ratio=99, extra={"policy": {"n_paths": 50_000}})
    pmf = effective_pmf(params.lead_time)
    x = np.arange(400)
    cdf = sum(q * np.cumsum(poisson_pmf(10.0 * (e + 1), 399)) for e, q in pmf.items())  # R = 1
    exact = int(x[np.argmax(cdf >= params.tau)])
    level = BaseStock(params, PoissonPaths(10.0)).level(make_state(0, pmf, seed=1, series_id="x"))
    assert abs(level - exact) <= 1, (level, exact)


@pytest.mark.parametrize("ratio", [9, 99])
def test_base_stock_uses_quantile_of_sum_not_sum_of_quantiles(ratio):
    params = make_params(ratio=ratio, lead=2)  # protection interval = 3 weeks
    tau = params.tau
    pmf = effective_pmf(params.lead_time)
    level = BaseStock(params, PoissonPaths(10.0)).level(make_state(0, pmf, seed=1, series_id="k"))

    exact = poisson_quantile(30.0, tau)
    summed = 3 * poisson_quantile(10.0, tau)
    assert abs(level - exact) <= 1
    assert level < summed - 3  # clearly below the forbidden sum-of-quantiles value
    expected = {9: (37, 42), 99: (43, 54)}[ratio]
    assert (exact, summed) == expected
