"""Forecast providers: joint demand sample paths (spec section 6.2).

Protection-interval demand is always computed by summing joint sample paths, never by summing
per-horizon quantiles. The interface therefore exposes no per-horizon quantile method.

These providers are test/oracle utilities. Real forecasters live in `forecasting/`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

import numpy as np


class ForecastProvider(Protocol):
    def sample_paths(
        self, series_id: str, origin: int, n_paths: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Joint demand paths, shape (n_paths, H), for weeks origin..origin+H-1.

        Must use only information through week origin-1.
        """
        ...


class OraclePaths:
    """Every path equals the true future demand (a perfect-information, zero-spread forecast).

    Weeks beyond the end of the series are padded with the last observed value, so a constant
    series stays constant. This peeks at the future by design; use only for validation tests.
    """

    def __init__(self, demand: np.ndarray | Mapping[str, np.ndarray], horizon: int = 8):
        self._demand = demand
        self.horizon = horizon

    def _series(self, series_id: str) -> np.ndarray:
        if isinstance(self._demand, Mapping):
            return np.asarray(self._demand[series_id], dtype=float)
        return np.asarray(self._demand, dtype=float)

    def sample_paths(
        self, series_id: str, origin: int, n_paths: int, rng: np.random.Generator
    ) -> np.ndarray:
        d = self._series(series_id)
        future = d[origin : origin + self.horizon]
        if len(future) < self.horizon:
            pad = np.full(self.horizon - len(future), d[-1])
            future = np.concatenate([future, pad])
        return np.tile(future, (n_paths, 1))


class PoissonPaths:
    """iid Poisson(lam) demand in every week, independent across paths (validation provider)."""

    def __init__(self, lam: float, horizon: int = 8):
        self.lam = lam
        self.horizon = horizon

    def sample_paths(
        self, series_id: str, origin: int, n_paths: int, rng: np.random.Generator
    ) -> np.ndarray:
        return rng.poisson(self.lam, size=(n_paths, self.horizon)).astype(float)
