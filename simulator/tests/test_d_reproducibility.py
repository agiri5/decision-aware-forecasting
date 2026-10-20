"""(d) Seeded reproducibility and common random numbers.

Proves: determinism, seed sensitivity, and that two policies with the same (seed, series_id)
face the identical supplier schedule. Does not prove: statistical quality of the RNG.
"""

import os
import subprocess
import sys
import textwrap

import numpy as np

from simulator import BaseStock, FixedOrderUpTo, PoissonPaths, simulate
from simulator.tests.helpers import stochastic_params

PARAMS = stochastic_params(shock_prob=0.2)
DEMAND = np.random.default_rng(7).poisson(8, size=120)


def _run(seed, series_id="x", policy=None):
    policy = policy or BaseStock(PARAMS, PoissonPaths(8.0))
    return simulate(DEMAND, policy, PARAMS, seed=seed, series_id=series_id)


def test_same_seed_identical():
    a, b = _run(3), _run(3)
    for name in ("ordered", "received", "on_hand_end", "lost", "arrival_week", "order_up_to"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name), err_msg=name)
    assert a.holding_cost.sum() == b.holding_cost.sum()


def test_different_seed_or_series_changes_lead_times():
    assert not np.array_equal(_run(1).lead_draws, _run(2).lead_draws)
    assert not np.array_equal(_run(1, "x").lead_draws, _run(1, "y").lead_draws)


def test_common_random_numbers_across_policies():
    a = _run(5, policy=FixedOrderUpTo(10))
    b = _run(5, policy=FixedOrderUpTo(60))
    c = _run(5)  # base-stock with Monte Carlo planning
    assert not np.array_equal(a.ordered, b.ordered)
    np.testing.assert_array_equal(a.arrival_week, b.arrival_week)
    np.testing.assert_array_equal(a.arrival_week, c.arrival_week)
    np.testing.assert_array_equal(a.lead_draws, c.lead_draws)


def test_independent_of_python_hash_seed():
    """Results must not depend on PYTHONHASHSEED (guards against using hash())."""
    script = textwrap.dedent(
        """
        import numpy as np
        from simulator import BaseStock, PoissonPaths, simulate
        from simulator.tests.helpers import stochastic_params
        p = stochastic_params(shock_prob=0.2)
        d = np.random.default_rng(7).poisson(8, size=60)
        r = simulate(d, BaseStock(p, PoissonPaths(8.0)), p, seed=3, series_id="x")
        print(int(r.ordered.sum()), int(r.lost.sum()), int(r.arrival_week.sum()))
        """
    )
    outs = []
    for hash_seed in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed}
        out = subprocess.run(
            [sys.executable, "-I", "-c", script],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        outs.append(out.stdout.strip())
    assert outs[0] == outs[1] != ""
