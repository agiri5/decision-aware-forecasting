"""(c) Unit conservation: the inventory identity holds every week.

Proves: no units are created or destroyed; cost arrays equal a recomputation from the unit
arrays. Does not prove: that the cost parameters themselves are sensible.
"""

import numpy as np
import pytest

from simulator import RandomOrders, simulate
from simulator.tests.helpers import stochastic_params


@pytest.mark.parametrize("mode", ["lost_sales", "backorder"])
@pytest.mark.parametrize("review", [1, 2])
@pytest.mark.parametrize("seed", range(5))
def test_inventory_identity(mode, review, seed):
    params = stochastic_params(mode=mode, review=review, shock_prob=0.3)
    rng = np.random.default_rng(100 + seed)
    demand = rng.poisson(5, size=300)
    res = simulate(
        demand,
        RandomOrders(rng, max_qty=15),
        params,
        seed=seed,
        series_id="cons",
        initial_on_hand=10,
    )

    prev_end = np.concatenate([[10], res.on_hand_end[:-1]])
    prev_pipe = np.concatenate([[0], res.pipeline_end[:-1]])
    np.testing.assert_array_equal(res.on_hand_start, prev_end)
    np.testing.assert_array_equal(res.on_hand_avail, res.on_hand_start + res.received)
    np.testing.assert_array_equal(res.on_hand_end, res.on_hand_avail - res.sold)
    np.testing.assert_array_equal(res.sold + res.lost, res.demand)
    np.testing.assert_array_equal(res.pipeline_end, prev_pipe + res.ordered - res.received)
    if mode == "lost_sales":
        assert np.all(res.on_hand_end >= 0) and np.all(res.lost >= 0)
    else:
        assert np.all(res.lost == 0)

    # Cumulative: OH_0 + sum(q) = OH_T + pipeline_T + sum(sold)
    assert 10 + res.ordered.sum() == res.on_hand_end[-1] + res.pipeline_end[-1] + res.sold.sum()
    assert res.received.sum() <= res.ordered.sum()

    # Costs recomputed from units (h = 1, p = 9, K = 0).
    np.testing.assert_allclose(res.holding_cost, np.maximum(res.on_hand_end, 0) * res.h)
    expected_short = res.lost if mode == "lost_sales" else np.maximum(-res.on_hand_end, 0)
    np.testing.assert_allclose(res.stockout_cost, expected_short * res.p)
    assert np.all(res.order_cost == 0)
