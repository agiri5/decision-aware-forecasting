"""Ordering policies (spec section 5).

BaseStock is the benchmark policy: order-up-to S, where S is the tau-quantile of Monte Carlo
protection-interval demand over (E + R) weeks, E ~ the effective lead-time pmf. tau = p/(p+h)
in the core design. The same heuristic is used for every forecaster. (s,S) is not built in v1
(docs/assumptions.md, D2).
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from simulator.leadtime import series_key
from simulator.params import SimParams
from simulator.providers import ForecastProvider
from simulator.state import State


class Policy(Protocol):
    """order() is called only on review weeks. Optional hooks, detected with getattr:

    reset()                 called at the start of each simulate() run
    initial_level(state0)   steady-state starting on-hand (e.g. the policy's S at t=0)
    """

    def order(self, state: State) -> int: ...


class BaseStock:
    def __init__(self, params: SimParams, provider: ForecastProvider):
        self.params = params
        self.provider = provider
        self.tau = params.tau
        self.n_paths = params.policy.n_paths
        self.review_period = params.simulation.review_period_weeks
        self.horizon = params.forecast.max_horizon_weeks
        self.levels: dict[int, int] = {}

    def reset(self) -> None:
        self.levels = {}

    def level(self, state: State) -> int:
        """S_t = ceil(tau-quantile of protection-interval demand), spec section 5.1.

        Pooled estimator: every sample path is summed over the protection interval for each
        effective lead time e in the pmf support (k = e + R weeks), and weighted by pmf(e) / N.
        This targets the same mixture distribution as drawing one E_n per path, without the
        lead-time sampling noise (G1 finding 1, D11).
        """
        n = self.n_paths
        # Child 0 of the planning stream is unused (it fed the per-path E_n draws before v1.1.0);
        # the provider keeps child 1, so its paths are unchanged.
        root = np.random.SeedSequence([state.seed, series_key(state.series_id), state.t, 1])
        _, ss_paths = root.spawn(2)
        paths = self.provider.sample_paths(
            state.series_id, state.t, n, np.random.default_rng(ss_paths)
        )
        if paths.shape != (n, self.horizon):
            raise ValueError(f"provider returned {paths.shape}, expected {(n, self.horizon)}")
        support = sorted(state.lead_time_pmf)
        probs = np.array([state.lead_time_pmf[e] for e in support])
        probs = probs / probs.sum()
        cum = np.cumsum(paths, axis=1)  # cum[:, k-1] = demand over the first k weeks
        sums = np.concatenate([cum[:, e + self.review_period - 1] for e in support])
        weights = np.repeat(probs / n, n)
        order = np.argsort(sums, kind="stable")
        cdf = np.cumsum(weights[order])
        # Smallest sample value with weighted CDF >= tau (the "inverted_cdf" rule); the small
        # tolerance absorbs float error in the cumulative weights.
        idx = min(int(np.searchsorted(cdf, self.tau - 1e-12, side="left")), len(sums) - 1)
        return int(np.ceil(sums[order][idx]))

    def initial_level(self, state0: State) -> int:
        return self.level(state0)

    def order(self, state: State) -> int:
        s_t = self.level(state)
        self.levels[state.t] = s_t
        return max(0, s_t - state.inventory_position)


class FixedOrderUpTo:
    """Test-only: constant order-up-to level S, no forecast."""

    def __init__(self, level: int):
        self.s = int(level)

    def initial_level(self, state0: State) -> int:
        return self.s

    def order(self, state: State) -> int:
        return max(0, self.s - state.inventory_position)


class ZeroPolicy:
    """Test-only: never orders."""

    def order(self, state: State) -> int:
        return 0


class RandomOrders:
    """Test-only: uniform random integer orders in [0, max_qty] (conservation tests)."""

    def __init__(self, rng: np.random.Generator, max_qty: int = 20):
        self.rng = rng
        self.max_qty = max_qty

    def order(self, state: State) -> int:
        return int(self.rng.integers(0, self.max_qty + 1))
