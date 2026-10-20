"""Lead-time schedule (common random numbers, no overtaking) and effective-lead-time pmf.

Spec section 4. Lead times are SYNTHETIC. They are pre-drawn per week (not per order) so every
policy run with the same (series_id, seed) faces the identical supplier schedule.
"""

from __future__ import annotations

import hashlib
import json
import zlib
from functools import lru_cache

import numpy as np

from simulator.params import LeadTimeParams

_PMF_BURN_IN = 100


def series_key(series_id: str) -> int:
    """Stable per-series integer for seeding. Never use hash() (salted per process)."""
    return zlib.crc32(series_id.encode())


def lead_time_rng(seed: int, series_id: str) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, series_key(series_id), 0]))


def draw_lead_times(lt: LeadTimeParams, n: int, rng: np.random.Generator) -> np.ndarray:
    """Per-week sampled lead times L_t (integer weeks), t = 0..n-1.

    L_t = clip(round(X_t) + shock_t * shock_weeks, min_weeks, max_weeks), X_t lognormal with
    the given mean/cv. Rounding is half-up. The rng stream layout is identical for every config
    (both draws always happen), so changing cv or shock_prob does not reshuffle other draws.
    """
    if lt.distribution == "deterministic":
        return np.full(n, lt.fixed_weeks, dtype=np.int64)
    sigma2 = np.log1p(lt.cv**2)
    mu = np.log(lt.mean_weeks) - sigma2 / 2.0
    x = rng.lognormal(mean=mu, sigma=np.sqrt(sigma2), size=n)
    shock = rng.random(n) < lt.shock_prob
    raw = np.floor(x + 0.5).astype(np.int64) + shock.astype(np.int64) * lt.shock_weeks
    return np.clip(raw, lt.min_weeks, lt.max_weeks)


def arrival_schedule(lead_draws: np.ndarray) -> np.ndarray:
    """A_t = max(t + L_t, A_{t-1}): non-decreasing by construction (no overtaking)."""
    t = np.arange(len(lead_draws), dtype=np.int64)
    return np.maximum.accumulate(t + lead_draws)


def lead_time_schedule(
    lt: LeadTimeParams, n: int, seed: int, series_id: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return (L_t draws, arrival weeks A_t) for weeks 0..n-1."""
    draws = draw_lead_times(lt, n, lead_time_rng(seed, series_id))
    return draws, arrival_schedule(draws)


@lru_cache(maxsize=64)
def _effective_pmf_cached(lt: LeadTimeParams) -> tuple[tuple[int, float], ...]:
    if lt.distribution == "deterministic":
        return ((lt.fixed_weeks, 1.0),)
    rng = np.random.default_rng(np.random.SeedSequence([lt.pmf_estimation_seed]))
    n = lt.pmf_estimation_weeks
    draws = draw_lead_times(lt, n, rng)
    eff = arrival_schedule(draws) - np.arange(n, dtype=np.int64)
    eff = eff[_PMF_BURN_IN:]
    support, counts = np.unique(eff, return_counts=True)
    probs = counts / counts.sum()
    return tuple((int(e), float(q)) for e, q in zip(support, probs, strict=True))


def effective_pmf(lt: LeadTimeParams) -> dict[int, float]:
    """Marginal pmf of the effective lead time E_t = A_t - t (after the no-overtaking max).

    Estimated once from one long run with a fixed seed; cached per lead-time params.
    """
    return dict(_effective_pmf_cached(lt))


def pmf_hash(pmf: dict[int, float]) -> str:
    blob = json.dumps(sorted((k, round(v, 12)) for k, v in pmf.items()))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]
