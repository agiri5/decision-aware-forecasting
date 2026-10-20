"""(g) No look-ahead, and policies cannot corrupt the simulation through State arrays."""

import numpy as np

from simulator import FixedOrderUpTo, simulate
from simulator.tests.helpers import make_params

DEMAND = np.random.default_rng(5).poisson(6, size=40)


class Spy:
    def __init__(self, tamper: bool):
        self.inner = FixedOrderUpTo(20)
        self.tamper = tamper
        self.seen = 0

    def initial_level(self, state0):
        return self.inner.initial_level(state0)

    def order(self, state):
        t = state.t
        assert len(state.demand_history) == t and len(state.sales_history) == t
        np.testing.assert_array_equal(state.demand_history, DEMAND[:t])
        self.seen += 1
        if self.tamper:
            state.demand_history[:] = 999
            state.sales_history[:] = 999
        return self.inner.order(state)


def test_history_is_prefix_only_and_tamper_proof():
    params = make_params(lead=2)
    honest, tamper = Spy(False), Spy(True)
    a = simulate(DEMAND, honest, params, seed=0, series_id="g")
    b = simulate(DEMAND, tamper, params, seed=0, series_id="g")
    assert honest.seen == tamper.seen == len(DEMAND)
    for name in ("ordered", "sold", "lost", "on_hand_end", "holding_cost"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name), err_msg=name)
