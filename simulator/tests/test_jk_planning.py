"""(j) Planning Monte Carlo and effective lead-time pmf; (k) no summed per-horizon quantiles."""

import pytest

from simulator import BaseStock, PoissonPaths, effective_pmf
from simulator.leadtime import lead_time_schedule
from simulator.tests.helpers import make_params, make_state, poisson_quantile


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


def test_mc_error_bounded_at_high_tau():
    params = make_params(ratio=99, lead=2)  # tau = 0.99, default n_paths
    pmf = effective_pmf(params.lead_time)
    levels = [
        BaseStock(params, PoissonPaths(10.0)).level(make_state(4, pmf, seed=s, series_id="mc"))
        for s in range(5)
    ]
    assert max(levels) - min(levels) <= 1, levels


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
