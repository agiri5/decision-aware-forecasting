"""(i) Arrivals never decrease (no overtaking), and receipts are FIFO at the scheduled week."""

import numpy as np
import pytest

from simulator import RandomOrders, simulate
from simulator.leadtime import lead_time_schedule
from simulator.tests.helpers import stochastic_params


@pytest.mark.parametrize("seed", range(50))
def test_schedule_is_monotone_and_never_same_week(seed):
    params = stochastic_params(shock_prob=0.3)
    draws, arrivals = lead_time_schedule(params.lead_time, 300, seed, "arr")
    t = np.arange(300)
    assert np.all(np.diff(arrivals) >= 0)
    assert np.all(arrivals >= t + 1)
    # Effective lead time recursion: E_t = max(L_t, E_{t-1} - 1)
    eff = arrivals - t
    assert np.all(eff[1:] == np.maximum(draws[1:], eff[:-1] - 1))
    assert eff.max() <= params.lead_time.max_weeks


@pytest.mark.parametrize("seed", range(5))
def test_receipts_land_exactly_at_scheduled_arrival(seed):
    params = stochastic_params(shock_prob=0.3)
    rng = np.random.default_rng(seed)
    res = simulate(
        rng.poisson(5, size=200),
        RandomOrders(rng, max_qty=10),
        params,
        seed=seed,
        series_id="fifo",
        initial_on_hand=5,
    )
    expected = np.zeros(len(res.demand), dtype=np.int64)
    for t, q in enumerate(res.ordered):
        arrival = res.arrival_week[t]
        if q > 0 and arrival < len(expected):
            expected[arrival] += q
    np.testing.assert_array_equal(res.received, expected)
